import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
import re
import os
import altair as alt
import pydeck as pdk 
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# =======================================================
# 1. TEXTOS LEGALES (MODELO DE NEGOCIO DE DATOS)
# =======================================================
TERMINOS_CONDICIONES = """
**TÉRMINOS Y CONDICIONES DE USO Y POLÍTICA DE DATOS - SMARTRECEIPT**

1. **ACEPTACIÓN:** Al acceder y utilizar esta plataforma, usted acepta estos términos en su totalidad.
2. **NATURALEZA DEL SERVICIO:** SmartReceipt utiliza Inteligencia Artificial para procesar tickets. El usuario reconoce que la IA puede cometer errores y es su responsabilidad verificar los montos.
3. **USO DE DATOS Y PROPIEDAD INTELECTUAL (CLÁUSULA DE NEGOCIO):** - Usted conserva la propiedad de sus tickets individuales.
   - Sin embargo, **usted otorga a SmartReceipt una licencia perpetua, irrevocable y mundial** para utilizar, copiar, modificar y agregar los datos procesados de forma **anónima** y **agregada** (sin identificarle personalmente) con fines de análisis estadístico, mejora del servicio, estudios de mercado y comercialización de insights de consumo.
4. **PRIVACIDAD:** Sus datos personales (nombre, correo) están protegidos. No vendemos su información de contacto a terceros.
5. **LIMITACIÓN DE RESPONSABILIDAD:** SmartReceipt no se hace responsable por pérdidas financieras derivadas de errores en la lectura de los tickets o fallos en el servicio.
"""

# =======================================================
# 2. CONFIGURACIÓN Y LOGIN
# =======================================================
st.set_page_config(page_title="SmartReceipt Enterprise", layout="wide", page_icon="🏢")

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""

def login():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><h1 style='text-align: center;'>🔐 Acceso Corporativo</h1>", unsafe_allow_html=True)
        with st.form("login_form"):
            usuario = st.text_input("Usuario")
            contra = st.text_input("Contraseña", type="password")
            st.markdown("---")
            with st.expander("📄 Leer Términos y Condiciones"): st.markdown(TERMINOS_CONDICIONES)
            acepto = st.checkbox("Acepto los Términos y Condiciones.")
            
            if st.form_submit_button("Ingresar", type="primary", use_container_width=True):
                if not acepto: st.error("Debe aceptar los términos.")
                elif "usuarios" in st.secrets and usuario in st.secrets["usuarios"]:
                    if st.secrets["usuarios"][usuario] == contra:
                        st.session_state.logged_in = True
                        st.session_state.username = usuario
                        st.rerun()
                    else: st.error("Contraseña incorrecta")
                else: st.error("Usuario no registrado")

if not st.session_state.logged_in:
    login()
    st.stop()

# =======================================================
# 3. GESTIÓN DE DATOS (CLOUD + LOCAL)
# =======================================================

# A) Configurar Gemini
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except: st.stop()

# B) Función para conectar a Sheets (Solo conecta, no descarga siempre)
def get_google_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            return client.open("SmartReceipt DB").sheet1
        return None
    except: return None

# C) Inicialización de Datos (Solo carga del Excel si la memoria está vacía)
if 'gastos' not in st.session_state or not st.session_state['gastos']:
    hoja = get_google_sheet()
    if hoja:
        try:
            # Auto-reparación headers
            if hoja.acell('A1').value != "Usuario":
                hoja.insert_row(["Usuario", "Fecha", "Comercio", "Monto", "Ubicación", "lat", "lon", "Categoría", "Detalles"], 1)
            
            raw = hoja.get_all_records()
            df_full = pd.DataFrame(raw)
            if not df_full.empty and "Usuario" in df_full.columns:
                # Filtrar solo lo de este usuario
                mis_gastos = df_full[df_full["Usuario"] == st.session_state.username].to_dict('records')
                st.session_state['gastos'] = mis_gastos
            else:
                st.session_state['gastos'] = []
        except: st.session_state['gastos'] = []
    else: st.session_state['gastos'] = []

if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

LISTA_CATEGORIAS = [
    "Alimentos y Supermercado", "Restaurantes y Bares", "Gasolina y Transporte",
    "Salud y Farmacia", "Hogar y Muebles", "Servicios (Luz/Agua/Internet)",
    "Telefonía y Comunicaciones", "Ropa y Calzado", "Electrónica y Tecnología",
    "Entretenimiento y Cine", "Educación y Libros", "Mascotas",
    "Regalos y Detalles", "Viajes y Hoteles", "Suscripciones (Streaming)",
    "Cuidado Personal y Belleza", "Deportes y Gimnasio", "Oficina y Trabajo",
    "Mantenimiento Automotriz", "Varios"
]

# =======================================================
# 4. FUNCIONES DE IA
# =======================================================
def procesar_imagen_opencv(imagen_pil):
    img_np = np.array(imagen_pil)
    if img_np.shape[-1] == 4: img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else: img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)

def analizar_ticket(imagen_pil):
    try:
        mods = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        modelo = next((m for m in mods if 'flash' in m and '1.5' in m), mods[0] if mods else "gemini-1.5-flash")
    except: modelo = "gemini-1.5-flash"
    
    try:
        model = genai.GenerativeModel(modelo)
        cats_str = ", ".join(LISTA_CATEGORIAS)
        prompt = f"""
        Analiza ticket. JSON EXCLUSIVO.
        UBICACIÓN: Busca SUCURSAL física y estima GPS (lat/lon).
        CATEGORÍA: [{cats_str}]
        JSON: {{"comercio": "Nombre", "total": 0.00, "fecha": "DD/MM/AAAA", "ubicacion": "Sucursal", "latitud": 19.0000, "longitud": -99.0000, "categoria": "Texto", "detalles": "Texto"}}
        """
        response = model.generate_content([prompt, imagen_pil])
        return response.text, modelo
    except Exception as e: return f"Error: {e}", modelo

def consultar_chat_financiero(pregunta, datos_df):
    # --- CORRECCIÓN APLICADA AQUÍ: Búsqueda dinámica de modelo para el chat ---
    try:
        mods = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        modelo = next((m for m in mods if 'flash' in m and '1.5' in m), mods[0] if mods else "gemini-1.5-flash")
    except: modelo = "gemini-1.5-flash"
    
    try:
        model = genai.GenerativeModel(modelo)
        datos_csv = datos_df.to_csv(index=False)
        prompt = f"Eres Asistente Financiero. Datos de {st.session_state.username}:\n---\n{datos_csv}\n---\nPregunta: {pregunta}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error Chat: {e}"

# =======================================================
# 5. INTERFAZ PRINCIPAL
# =======================================================
with st.sidebar:
    st.header(f"👤 {st.session_state.username}")
    
    # BOTÓN DE SINCRONIZACIÓN FORZADA (Por si acaso)
    if st.button("🔄 Forzar Sincronización Nube"):
        st.cache_data.clear()
        if 'gastos' in st.session_state: del st.session_state['gastos'] 
        st.rerun()
        
    if st.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.rerun()
    st.divider()
    
    # Filtros sobre la memoria local
    df_local = pd.DataFrame(st.session_state['gastos'])
    df_filtrado = pd.DataFrame()
    
    if not df_local.empty:
        # Asegurar tipos
        for c in ['lat','lon','Monto']:
            if c in df_local.columns: df_local[c] = pd.to_numeric(df_local[c], errors='coerce').fillna(0.0)
            
        cat_opts = sorted(df_local['Categoría'].astype(str).unique()) if 'Categoría' in df_local.columns else []
        sel_cat = st.sidebar.multiselect("Categoría", cat_opts)
        
        if sel_cat: df_filtrado = df_local[df_local['Categoría'].isin(sel_cat)]
        else: df_filtrado = df_local

st.title("💳 SmartReceipt: Enterprise")
tab_nuevo, tab_dashboard, tab_chat = st.tabs(["📸 Nuevo Ticket", "📈 Analytics", "💬 Asistente IA"])

# --- TAB 1: CARGA ---
with tab_nuevo:
    col1, col2 = st.columns(2)
    with col1:
        archivo = st.file_uploader("Sube ticket", type=["jpg","png","jpeg"])
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, caption="Ticket", use_container_width=True)
            if st.button("⚡ Procesar", type="primary"):
                with st.spinner("Leyendo..."):
                    txt, mod = analizar_ticket(img_proc)
                    if "Error" in txt: st.error(txt)
                    else:
                        try:
                            match = re.search(r'\{.*\}', txt, re.DOTALL)
                            if match:
                                st.session_state['temp_data'] = json.loads(match.group())
                                st.toast("Leído")
                        except: st.error("Error lectura")

    with col2:
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.form("save_form"):
                st.subheader("Validar")
                c1,c2 = st.columns(2)
                vc = c1.text_input("Comercio", data.get("comercio",""))
                vm = c2.number_input("Total", value=float(str(data.get("total",0)).replace("$","").replace(",","")))
                c3,c4 = st.columns(2)
                vf = c3.text_input("Fecha", data.get("fecha",""))
                cat_def = data.get("categoria","Varios")
                idx = LISTA_CATEGORIAS.index(cat_def) if cat_def in LISTA_CATEGORIAS else 19
                vcat = c4.selectbox("Categoría", LISTA_CATEGORIAS, index=idx)
                vu = st.text_input("Sucursal", data.get("ubicacion",""))
                vdet = st.text_input("Detalles", data.get("detalles",""))
                vlat = float(data.get("latitud", 0.0))
                vlon = float(data.get("longitud", 0.0))

                if st.form_submit_button("💾 Guardar (Instantáneo)"):
                    # 1. ACTUALIZACIÓN OPTIMISTA (LOCAL)
                    nuevo_registro = {
                        "Usuario": st.session_state.username,
                        "Fecha": vf, "Comercio": vc, "Monto": vm, 
                        "Ubicación": vu, "lat": vlat, "lon": vlon,
                        "Categoría": vcat, "Detalles": vdet
                    }
                    st.session_state['gastos'].append(nuevo_registro)
                    
                    # 2. ACTUALIZACIÓN NUBE (BACKGROUND)
                    hoja = get_google_sheet()
                    if hoja:
                        try:
                            nueva_fila = [st.session_state.username, vf, vc, vm, vu, vlat, vlon, vcat, vdet]
                            hoja.append_row(nueva_fila)
                            st.success("Guardado y Sincronizado.")
                        except:
                            st.warning("Guardado localmente. Error sincronizando nube.")
                    
                    del st.session_state['temp_data']
                    st.rerun() # Recarga para actualizar gráficas al instante

# --- TAB 2: DASHBOARD ---
with tab_dashboard:
    if not df_filtrado.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Gastado", f"${df_filtrado['Monto'].sum():,.2f}")
        c2.metric("Tickets", len(df_filtrado))
        c3.metric("Promedio", f"${df_filtrado['Monto'].mean():,.2f}")
        st.divider()
        
        map_data = df_filtrado[(df_filtrado['lat']!=0)]
        if not map_data.empty:
            st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=11),
                layers=[pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_color='[255, 75, 75, 200]', get_radius=200, pickable=True)],
                tooltip={"html": "<b>{Comercio}</b><br/>${Monto}"}))
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.altair_chart(alt.Chart(df_filtrado).mark_arc(innerRadius=50).encode(theta='Monto', color='Categoría', tooltip=['Categoría','Monto']), use_container_width=True)
        with col_g2:
             st.dataframe(df_filtrado, use_container_width=True, height=300)
    else: st.info("Sin datos.")

# --- TAB 3: CHAT ---
with tab_chat:
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("Pregunta..."):
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role":"user", "content":q})
        if df_filtrado.empty: r = "Sin datos."
        else:
            with st.spinner("..."): r = consultar_chat_financiero(q, df_filtrado)
        with st.chat_message("assistant"): st.markdown(r)
        st.session_state['chat_history'].append({"role":"assistant", "content":r})

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© 2025 SmartReceipt Inc. | Privacidad y Términos</div>", unsafe_allow_html=True)