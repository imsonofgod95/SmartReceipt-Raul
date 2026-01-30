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

# =======================================================
# 1. CONFIGURACIÓN SEGURA & CONEXIÓN A SHEETS ☁️
# =======================================================
st.set_page_config(page_title="SmartReceipt Pro", layout="wide", page_icon="📈")

# A) Configurar Gemini (IA)
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("⚠️ Falta GOOGLE_API_KEY en Secrets.")
        st.stop()
except Exception as e:
    st.error(f"Error config Gemini: {e}")
    st.stop()

# B) Configurar Google Sheets (Base de Datos)
def conectar_google_sheets():
    """Conecta a la hoja de cálculo usando las credenciales de Secrets"""
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    try:
        # Leemos las credenciales desde [gcp_service_account] en Secrets
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            
            # Arreglar el formato de la private_key si es necesario
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            
            # Abrimos la hoja por su nombre exacto
            sheet = client.open("SmartReceipt DB").sheet1
            return sheet
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error conectando a Google Sheets: {e}")
        return None

# Cargar datos al inicio
try:
    hoja_db = conectar_google_sheets()
    if hoja_db:
        raw_data = hoja_db.get_all_records()
        df_gastos = pd.DataFrame(raw_data)
        
        # Si la hoja está vacía (solo headers), creamos un DF vacío con columnas correctas
        if df_gastos.empty:
            df_gastos = pd.DataFrame(columns=["Fecha", "Comercio", "Monto", "Ubicación", "lat", "lon", "Categoría", "Detalles"])
    else:
        df_gastos = pd.DataFrame()
except:
    # st.warning("Modo Offline: No se pudo conectar a Sheets.")
    df_gastos = pd.DataFrame()

# Guardar en Session State para la UI
if 'gastos' not in st.session_state:
    st.session_state['gastos'] = df_gastos.to_dict('records')

# LISTA MAESTRA DE CATEGORÍAS
LISTA_CATEGORIAS = [
    "Alimentos y Supermercado", "Restaurantes y Bares", "Gasolina y Transporte",
    "Salud y Farmacia", "Hogar y Muebles", "Servicios (Luz/Agua/Internet)",
    "Telefonía y Comunicaciones", "Ropa y Calzado", "Electrónica y Tecnología",
    "Entretenimiento y Cine", "Educación y Libros", "Mascotas",
    "Regalos y Detalles", "Viajes y Hoteles", "Suscripciones (Streaming)",
    "Cuidado Personal y Belleza", "Deportes y Gimnasio", "Oficina y Trabajo",
    "Mantenimiento Automotriz", "Varios"
]

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# =======================================================
# 2. MÓDULO DE VISIÓN
# =======================================================
def procesar_imagen_opencv(imagen_pil):
    img_np = np.array(imagen_pil)
    if img_np.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)

# =======================================================
# 3. CONEXIÓN IA (DINÁMICA - BUSCA EL MODELO DISPONIBLE)
# =======================================================
def obtener_modelo_valido():
    """
    Busca dinámicamente qué modelos tiene habilitados la API Key
    y selecciona el mejor disponible (Flash > Pro > Cualquiera).
    """
    try:
        # Obtenemos lista real de modelos disponibles para tu cuenta
        modelos = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1. Buscar Flash 1.5 (Prioridad)
        for m in modelos:
            if 'flash' in m and '1.5' in m: return m
        
        # 2. Buscar Pro 1.5
        for m in modelos:
            if 'pro' in m and '1.5' in m: return m
            
        # 3. Fallback: Pro 1.0 o el primero que encuentre
        for m in modelos:
            if 'pro' in m: return m
            
        return modelos[0] if modelos else "gemini-1.5-flash"
    except Exception as e:
        # Si falla el listado, intentamos el default ciegamente
        return "gemini-1.5-flash"

def analizar_ticket(imagen_pil):
    nombre_modelo = obtener_modelo_valido()
    try:
        model = genai.GenerativeModel(nombre_modelo)
        cats_str = ", ".join(LISTA_CATEGORIAS)
        prompt = f"""
        Analiza ticket. JSON EXCLUSIVO.
        UBICACIÓN: Busca SUCURSAL física y estima GPS (lat/lon).
        CATEGORÍA: [{cats_str}]
        
        JSON:
        {{
            "comercio": "Nombre", "total": 0.00, "fecha": "DD/MM/AAAA",
            "ubicacion": "Sucursal", "latitud": 19.0000, "longitud": -99.0000,
            "categoria": "Texto", "detalles": "Texto"
        }}
        """
        response = model.generate_content([prompt, imagen_pil])
        return response.text, nombre_modelo
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg: return "CUOTA_EXCEDIDA: Espera un momento.", nombre_modelo
        if "404" in error_msg: return f"ERROR_MODELO: No se encontró el modelo {nombre_modelo}. Actualiza requirements.txt", nombre_modelo
        if "API key" in error_msg: return "ERROR_KEY: Llave inválida.", nombre_modelo
        return f"Error Técnico: {e}", nombre_modelo

def consultar_chat_financiero(pregunta, datos_df):
    nombre = obtener_modelo_valido()
    try:
        model = genai.GenerativeModel(nombre)
        datos_csv = datos_df.to_csv(index=False)
        prompt_sistema = f"""
        Eres un Asistente Financiero. Datos:
        ---
        {datos_csv}
        ---
        Pregunta: {pregunta}
        """
        response = model.generate_content(prompt_sistema)
        return response.text
    except Exception as e:
        return f"Error Chat: {e}"

# =======================================================
# 4. DASHBOARD & FILTROS
# =======================================================
st.sidebar.header("🔍 Filtros")
df = pd.DataFrame(st.session_state['gastos'])
df_filtrado = pd.DataFrame()

if not df.empty:
    # Asegurar tipos numéricos para evitar errores
    if 'lat' in df.columns: df['lat'] = pd.to_numeric(df['lat'], errors='coerce').fillna(0.0)
    if 'lon' in df.columns: df['lon'] = pd.to_numeric(df['lon'], errors='coerce').fillna(0.0)
    if 'Monto' in df.columns: df['Monto'] = pd.to_numeric(df['Monto'], errors='coerce').fillna(0.0)

    # Filtros
    cat_opts = sorted(df['Categoría'].astype(str).unique()) if 'Categoría' in df.columns else []
    com_opts = sorted(df['Comercio'].astype(str).unique()) if 'Comercio' in df.columns else []
    
    sel_cat = st.sidebar.multiselect("Categoría", cat_opts, default=cat_opts)
    sel_com = st.sidebar.multiselect("Comercio", com_opts, default=com_opts)
    
    mask_cat = df['Categoría'].isin(sel_cat) if 'Categoría' in df.columns else True
    mask_com = df['Comercio'].isin(sel_com) if 'Comercio' in df.columns else True
    df_filtrado = df[mask_cat & mask_com]
else:
    st.sidebar.info("Base de datos vacía. Sube tu primer ticket.")

# =======================================================
# 5. UI PRINCIPAL
# =======================================================
st.title("💳 SmartReceipt: Business Cloud")
st.markdown("---")

tab_nuevo, tab_dashboard, tab_chat = st.tabs(["📸 Nuevo Ticket", "📈 Analytics", "💬 Asistente IA"])

# --- TAB 1: CARGA (CON DIAGNÓSTICO MEJORADO) ---
with tab_nuevo:
    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        archivo = st.file_uploader("Sube ticket", type=["jpg", "png", "jpeg"])
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, caption="Ticket Procesado", use_container_width=True)
            
            if st.button("⚡ Procesar", type="primary"):
                with st.spinner("Conectando con Google AI..."):
                    txt, mod = analizar_ticket(img_proc)
                    
                    # --- BLOQUE DE DIAGNÓSTICO ---
                    if "Error" in txt or "CUOTA" in txt or "ERROR" in txt:
                        st.error(f"🛑 {txt}")
                        if "404" in txt:
                            st.info("💡 Solución: Ve a tu archivo 'requirements.txt' en GitHub y cambia 'google-generativeai' por 'google-generativeai>=0.7.0'")
                    else:
                        try:
                            match = re.search(r'\{.*\}', txt, re.DOTALL)
                            if match:
                                clean_json = match.group()
                                st.session_state['temp_data'] = json.loads(clean_json)
                                st.toast(f"Leído con {mod}", icon="✨")
                            else:
                                st.error("⚠️ La IA no devolvió datos legibles.")
                                with st.expander("Ver respuesta cruda (Debug)"):
                                    st.code(txt)
                        except Exception as e: 
                            st.error(f"Error procesando JSON: {e}")

    with col_der:
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.form("form_save"):
                st.subheader("Confirmar y Guardar en Nube")
                c1, c2 = st.columns(2)
                vc = c1.text_input("Comercio", data.get("comercio",""))
                monto_raw = str(data.get("total",0)).replace("$","").replace(",","")
                try: vm = float(monto_raw)
                except: vm = 0.0
                vm_in = c2.number_input("Total", value=vm)
                
                c3, c4 = st.columns(2)
                vf = c3.text_input("Fecha", data.get("fecha","Hoy"))
                cat_in = data.get("categoria", "Varios")
                idx = LISTA_CATEGORIAS.index(cat_in) if cat_in in LISTA_CATEGORIAS else 19
                vcat = c4.selectbox("Categoría", LISTA_CATEGORIAS, index=idx)
                
                c5, c6, c7 = st.columns([2, 1, 1])
                vu = c5.text_input("Sucursal", data.get("ubicacion",""))
                vlat = c6.number_input("Lat", value=float(data.get("latitud", 0.0)), format="%.4f")
                vlon = c7.number_input("Lon", value=float(data.get("longitud", 0.0)), format="%.4f")
                vdet = st.text_input("Detalles", data.get("detalles",""))
                
                if st.form_submit_button("☁️ Guardar en Google Sheets"):
                    with st.spinner("Guardando en la nube..."):
                        # Preparar fila para Sheets
                        nueva_fila = [vf, vc, vm_in, vu, vlat, vlon, vcat, vdet]
                        
                        # Guardar en Sheets
                        if hoja_db:
                            try:
                                hoja_db.append_row(nueva_fila)
                                st.success("¡Guardado en Drive!")
                                # Actualizar estado local para que se vea inmediato
                                st.session_state['gastos'].append({
                                    "Fecha": vf, "Comercio": vc, "Monto": vm_in, 
                                    "Ubicación": vu, "lat": vlat, "lon": vlon,
                                    "Categoría": vcat, "Detalles": vdet
                                })
                                del st.session_state['temp_data']
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error escribiendo en Sheets: {e}")
                        else:
                            st.error("No hay conexión con la base de datos.")

# --- TAB 2: DASHBOARD ---
with tab_dashboard:
    if not df_filtrado.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("💰 Gastado", f"${df_filtrado['Monto'].sum():,.2f}")
        k2.metric("📊 Promedio", f"${df_filtrado['Monto'].mean():,.2f}")
        k3.metric("🧾 Tickets", len(df_filtrado))
        st.divider()
        
        map_data = df_filtrado[(df_filtrado['lat']!=0) & (df_filtrado['lon']!=0)]
        if not map_data.empty:
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=11, pitch=40),
                layers=[pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_color='[255, 75, 75, 200]', get_radius=200, pickable=True)],
                tooltip={"html": "<b>{Comercio}</b><br/>${Monto}"}
            ))
        else: st.info("Sube tickets con ubicación para ver mapa.")
        
        g1, g2 = st.columns(2)
        with g1:
            st.altair_chart(alt.Chart(df_filtrado).mark_arc(innerRadius=60).encode(theta='Monto', color='Categoría', tooltip=['Categoría', 'Monto']), use_container_width=True)
        with g2:
            df_chart = df_filtrado.copy()
            df_chart['Fecha_dt'] = pd.to_datetime(df_chart['Fecha'], dayfirst=True, errors='coerce')
            df_chart = df_chart.dropna(subset=['Fecha_dt']).sort_values('Fecha_dt')
            if not df_chart.empty:
                st.altair_chart(alt.Chart(df_chart).mark_line(point=True).encode(x=alt.X('Fecha_dt', title='Fecha'), y='Monto', tooltip=['Fecha', 'Monto']), use_container_width=True)

        with st.expander("📂 Ver Base de Datos en Vivo (Google Sheets)"):
            st.dataframe(df_filtrado, use_container_width=True)
            if st.button("🔄 Recargar Datos de la Nube"):
                st.cache_data.clear()
                st.rerun()

# --- TAB 3: CHAT ---
with tab_chat:
    st.header("💬 Analista Financiero")
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    q = st.chat_input("Pregunta...")
    if q:
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role": "user", "content": q})
        if df_filtrado.empty: r = "Sube tickets primero."
        else:
            with st.spinner("Analizando..."):
                r = consultar_chat_financiero(q, pd.DataFrame(st.session_state['gastos']))
        with st.chat_message("assistant"): st.markdown(r)
        st.session_state['chat_history'].append({"role": "assistant", "content": r})