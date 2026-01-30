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
# 1. TEXTOS LEGALES ROBUSTOS (MODELO DE NEGOCIO DE DATOS)
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
# 2. CONFIGURACIÓN Y LOGIN CON CHECK LEGAL ✅
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
            # ACUERDO LEGAL OBLIGATORIO
            with st.expander("📄 Leer Términos y Condiciones de Uso de Datos"):
                st.markdown(TERMINOS_CONDICIONES)
            
            acepto_terminos = st.checkbox("He leído y acepto los Términos y el uso de datos anónimos.")
            
            if st.form_submit_button("Ingresar al Sistema", type="primary", use_container_width=True):
                if not acepto_terminos:
                    st.error("🛑 Debe aceptar los términos legales para continuar.")
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
# 3. SINCRONIZACIÓN AUTOMÁTICA (EL SECRETO DEL ÉXITO) 🔄
# =======================================================

# A) Configurar Gemini
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except: st.stop()

# B) Función de Carga Maestra
def sincronizar_base_datos():
    """Conecta, repara y descarga la última versión de los datos"""
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            hoja = client.open("SmartReceipt DB").sheet1
            
            # Auto-reparación silenciosa
            val_a1 = hoja.acell('A1').value
            if val_a1 != "Usuario":
                encabezados = ["Usuario", "Fecha", "Comercio", "Monto", "Ubicación", "lat", "lon", "Categoría", "Detalles"]
                if not val_a1: hoja.append_row(encabezados)
                else: hoja.insert_row(encabezados, 1)
            
            return hoja, pd.DataFrame(hoja.get_all_records())
        return None, pd.DataFrame()
    except Exception as e:
        return None, pd.DataFrame()

# C) EJECUCIÓN INMEDIATA (ESTO HACE QUE CARGUE SIEMPRE)
hoja_db, df_full = sincronizar_base_datos()

# D) FILTRADO DE SEGURIDAD
if not df_full.empty and "Usuario" in df_full.columns:
    df_gastos = df_full[df_full["Usuario"] == st.session_state.username].copy()
    
    # Limpieza de tipos de datos para evitar errores en gráficas
    for col in ['lat', 'lon', 'Monto']:
        if col in df_gastos.columns:
            df_gastos[col] = pd.to_numeric(df_gastos[col], errors='coerce').fillna(0.0)
else:
    df_gastos = pd.DataFrame(columns=["Usuario", "Fecha", "Comercio", "Monto", "Ubicación", "lat", "lon", "Categoría", "Detalles"])

# E) Guardar en Session State
st.session_state['gastos'] = df_gastos.to_dict('records')

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
# 4. FUNCIONES DE PROCESAMIENTO
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
    except Exception as e:
        return f"Error: {e}", modelo

def consultar_chat_financiero(pregunta, datos_df):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        datos_csv = datos_df.to_csv(index=False)
        prompt = f"Eres Asistente Financiero. Datos: \n---\n{datos_csv}\n---\nPregunta: {pregunta}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error Chat: {e}"

# =======================================================
# 5. INTERFAZ PRINCIPAL (UI)
# =======================================================
with st.sidebar:
    st.header(f"👤 {st.session_state.username}")
    st.caption("🟢 Conexión Segura Activa")
    
    if st.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.rerun()
    st.divider()
    
    # Filtros
    df_filtrado = df_gastos.copy() # Usamos la copia fresca cargada al inicio
    if not df_filtrado.empty:
        cat_opts = sorted(df_filtrado['Categoría'].astype(str).unique()) if 'Categoría' in df_filtrado.columns else []
        sel_cat = st.sidebar.multiselect("Categoría", cat_opts)
        if sel_cat: df_filtrado = df_filtrado[df_filtrado['Categoría'].isin(sel_cat)]

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
            st.image(img_proc, caption="Vista Previa", use_container_width=True)
            if st.button("⚡ Procesar", type="primary"):
                with st.spinner("Analizando..."):
                    txt, mod = analizar_ticket(img_proc)
                    if "Error" in txt: st.error(txt)
                    else:
                        try:
                            match = re.search(r'\{.*\}', txt, re.DOTALL)
                            if match:
                                st.session_state['temp_data'] = json.loads(match.group())
                                st.toast("Datos extraídos")
                        except: st.error("Error leyendo datos")

    with col2:
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.form("save_form"):
                st.subheader("Validar Datos")
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

                if st.form_submit_button("💾 Guardar y Sincronizar"):
                    nueva_fila = [st.session_state.username, vf, vc, vm, vu, vlat, vlon, vcat, vdet]
                    if hoja_db:
                        try:
                            hoja_db.append_row(nueva_fila)
                            st.success("Guardado exitoso.")
                            # TRUCO: Al hacer rerun, el script vuelve al inicio y ejecuta 
                            # 'sincronizar_base_datos()' automáticamente, actualizando todo.
                            st.rerun()
                        except: st.error("Error conexión DB")

# --- TAB 2: DASHBOARD ---
with tab_dashboard:
    if not df_filtrado.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Gastado", f"${df_filtrado['Monto'].sum():,.2f}")
        c2.metric("Tickets Procesados", len(df_filtrado))
        c3.metric("Ticket Promedio", f"${df_filtrado['Monto'].mean():,.2f}")
        st.divider()
        
        map_data = df_filtrado[(df_filtrado['lat']!=0)]
        if not map_data.empty:
            st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=11),
                layers=[pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_color='[255, 75, 75, 200]', get_radius=200, pickable=True)],
                tooltip={"html": "<b>{Comercio}</b><br/>${Monto}"}))
        
        st.altair_chart(alt.Chart(df_filtrado).mark_arc(innerRadius=50).encode(theta='Monto', color='Categoría', tooltip=['Categoría','Monto']), use_container_width=True)
        st.dataframe(df_filtrado, use_container_width=True)
    else: st.info("No hay datos visibles.")

# --- TAB 3: CHAT ---
with tab_chat:
    st.caption("🤖 Pregunta sobre tus gastos. La IA tiene acceso a tus datos actualizados.")
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("Ej: ¿Cuánto gasté en comida este mes?"):
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role":"user", "content":q})
        if df_filtrado.empty: r = "No tienes datos aún."
        else:
            with st.spinner("Analizando..."): r = consultar_chat_financiero(q, df_filtrado)
        with st.chat_message("assistant"): st.markdown(r)
        st.session_state['chat_history'].append({"role":"assistant", "content":r})