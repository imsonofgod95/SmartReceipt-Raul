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
# 1. CONFIGURACIÓN Y ESTILOS PREMIUM (CSS) 🎨
# =======================================================
st.set_page_config(page_title="SmartReceipt Enterprise", layout="wide", page_icon="💳")

# --- DISEÑO UI/UX PROFESIONAL ---
st.markdown("""
    <style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Degradado en el encabezado principal */
    .main-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        padding-bottom: 10px;
    }

    /* Estilo de Tarjetas (Métricas) */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Botones más profesionales */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
        width: 100%;
    }

    /* Footer fijo y estilizado */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        z-index: 999;
    }
    
    /* Sidebar más limpio */
    [data-testid="stSidebar"] {
        background-color: #F3F4F6;
        border-right: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# =======================================================
# 2. LÓGICA DE LOGIN
# =======================================================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""

def login():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        # Logo o Título grande
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏢 SmartReceipt <span style='font-size: 0.5em; color: #64748B;'>Enterprise</span></h1>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("### Iniciar Sesión")
            with st.form("login_form"):
                usuario = st.text_input("Usuario", placeholder="ej. raul")
                contra = st.text_input("Contraseña", type="password", placeholder="••••••")
                
                st.caption("Al continuar, aceptas nuestros Términos de Servicio y Política de Privacidad.")
                acepto = st.checkbox("Acepto los términos legales.")
                
                if st.form_submit_button("Acceder al Dashboard", type="primary", use_container_width=True):
                    if not acepto: st.warning("⚠️ Debes aceptar los términos.")
                    elif "usuarios" in st.secrets and usuario in st.secrets["usuarios"]:
                        if st.secrets["usuarios"][usuario] == contra:
                            st.session_state.logged_in = True
                            st.session_state.username = usuario
                            st.rerun()
                        else: st.error("Contraseña incorrecta")
                    else: st.error("Usuario no encontrado")

if not st.session_state.logged_in:
    login()
    st.stop()

# =======================================================
# 3. BACKEND Y CONEXIONES 🧠
# =======================================================
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except: st.stop()

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

# Inicialización optimizada
if 'gastos' not in st.session_state or not st.session_state['gastos']:
    hoja = get_google_sheet()
    if hoja:
        try:
            if hoja.acell('A1').value != "Usuario":
                hoja.insert_row(["Usuario", "Fecha", "Comercio", "Monto", "Ubicación", "lat", "lon", "Categoría", "Detalles"], 1)
            raw = hoja.get_all_records()
            df_full = pd.DataFrame(raw)
            if not df_full.empty and "Usuario" in df_full.columns:
                mis_gastos = df_full[df_full["Usuario"] == st.session_state.username].to_dict('records')
                st.session_state['gastos'] = mis_gastos
            else: st.session_state['gastos'] = []
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
# 4. FUNCIONES CORE
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
# 5. DASHBOARD PRINCIPAL
# =======================================================

# --- SIDEBAR MEJORADO ---
with st.sidebar:
    st.markdown(f"""
    <div style="background-color: #E0E7FF; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0; color: #1E3A8A;">👤 {st.session_state.username}</h3>
        <p style="margin:0; font-size: 12px; color: #6B7280;">Cuenta Enterprise</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Sincronizar Nube"):
        st.cache_data.clear()
        if 'gastos' in st.session_state: del st.session_state['gastos'] 
        st.rerun()
        
    if st.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.rerun()
        
    st.markdown("### 📊 Filtros Inteligentes")
    df_local = pd.DataFrame(st.session_state['gastos'])
    df_filtrado = pd.DataFrame()
    
    if not df_local.empty:
        for c in ['lat','lon','Monto']:
            if c in df_local.columns: df_local[c] = pd.to_numeric(df_local[c], errors='coerce').fillna(0.0)
        cat_opts = sorted(df_local['Categoría'].astype(str).unique()) if 'Categoría' in df_local.columns else []
        sel_cat = st.sidebar.multiselect("Categoría", cat_opts)
        df_filtrado = df_local[df_local['Categoría'].isin(sel_cat)] if sel_cat else df_local

# --- HEADER PRINCIPAL ---
st.markdown('<h1 class="main-header">SmartReceipt <span style="font-weight:300;">Enterprise</span></h1>', unsafe_allow_html=True)

# --- METRICAS FLOTANTES (ESTILO TARJETAS) ---
if not df_filtrado.empty:
    m1, m2, m3 = st.columns(3)
    
    # Inyección HTML directa para tarjetas de métricas bonitas
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1rem; color: #64748B;">Gasto Total</h3>
            <h2 style="margin:0; font-size: 2rem; color: #1E3A8A;">${df_filtrado['Monto'].sum():,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1rem; color: #64748B;">Tickets</h3>
            <h2 style="margin:0; font-size: 2rem; color: #059669;">{len(df_filtrado)}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1rem; color: #64748B;">Promedio</h3>
            <h2 style="margin:0; font-size: 2rem; color: #D97706;">${df_filtrado['Monto'].mean():,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

# --- TABS CON ESTILO ---
tab_nuevo, tab_dashboard, tab_chat = st.tabs(["📸 Nuevo Ticket", "📈 Analytics", "💬 Asistente IA"])

# TAB 1
with tab_nuevo:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### 1. Sube tu imagen")
        archivo = st.file_uploader("Arrastra tu ticket aquí", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, caption="Imagen Optimizada", use_container_width=True)
            if st.button("⚡ Procesar con IA", type="primary"):
                with st.spinner("Extrayendo datos financieros..."):
                    txt, mod = analizar_ticket(img_proc)
                    if "Error" in txt: st.error(txt)
                    else:
                        try:
                            match = re.search(r'\{.*\}', txt, re.DOTALL)
                            if match:
                                st.session_state['temp_data'] = json.loads(match.group())
                                st.toast("Lectura exitosa", icon="✅")
                        except: st.error("Error leyendo datos")

    with col2:
        st.markdown("#### 2. Verifica y Guarda")
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.container(border=True):
                c1,c2 = st.columns(2)
                vc = c1.text_input("Comercio", data.get("comercio",""))
                vm = c2.number_input("Total ($)", value=float(str(data.get("total",0)).replace("$","").replace(",","")))
                c3,c4 = st.columns(2)
                vf = c3.text_input("Fecha", data.get("fecha",""))
                cat_def = data.get("categoria","Varios")
                idx = LISTA_CATEGORIAS.index(cat_def) if cat_def in LISTA_CATEGORIAS else 19
                vcat = c4.selectbox("Categoría", LISTA_CATEGORIAS, index=idx)
                
                with st.expander("📍 Datos de Ubicación y Detalles"):
                    vu = st.text_input("Sucursal", data.get("ubicacion",""))
                    vdet = st.text_input("Detalles", data.get("detalles",""))
                    vlat = float(data.get("latitud", 0.0))
                    vlon = float(data.get("longitud", 0.0))

                if st.button("💾 Guardar en Nube", type="primary", use_container_width=True):
                    nuevo = {"Usuario": st.session_state.username, "Fecha": vf, "Comercio": vc, "Monto": vm, "Ubicación": vu, "lat": vlat, "lon": vlon, "Categoría": vcat, "Detalles": vdet}
                    st.session_state['gastos'].append(nuevo)
                    hoja = get_google_sheet()
                    if hoja:
                        try:
                            hoja.append_row(list(nuevo.values()))
                            st.success("Ticket registrado correctamente.")
                        except: st.warning("Guardado local (Sin conexión a Sheets).")
                    del st.session_state['temp_data']
                    st.rerun()
        else:
            st.info("Esperando imagen para mostrar formulario...")

# TAB 2
with tab_dashboard:
    if not df_filtrado.empty:
        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            st.markdown("##### Distribución por Categoría")
            st.altair_chart(alt.Chart(df_filtrado).mark_arc(innerRadius=60).encode(theta='Monto', color=alt.Color('Categoría', scale={'scheme': 'tableau10'}), tooltip=['Categoría','Monto']), use_container_width=True)
        with col_g2:
            st.markdown("##### Historial de Transacciones")
            st.dataframe(df_filtrado, use_container_width=True, height=400, hide_index=True)
            
        map_data = df_filtrado[(df_filtrado['lat']!=0)]
        if not map_data.empty:
            st.markdown("##### 🗺️ Mapa de Consumo")
            st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=11),
                layers=[pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_color='[37, 99, 235, 180]', get_radius=200, pickable=True)],
                tooltip={"html": "<b>{Comercio}</b><br/>${Monto}"}))
    else: st.warning("No hay datos para mostrar.")

# TAB 3
with tab_chat:
    st.info("🤖 **AI Financial Advisor:** Pregunta sobre tus tendencias, totales o busca tickets específicos.")
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("Ej: ¿En qué gasté más dinero este mes?"):
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role":"user", "content":q})
        if df_filtrado.empty: r = "Sin datos."
        else:
            with st.spinner("Analizando finanzas..."): r = consultar_chat_financiero(q, df_filtrado)
        with st.chat_message("assistant"): st.markdown(r)
        st.session_state['chat_history'].append({"role":"assistant", "content":r})

# FOOTER
st.markdown("""
<div class="footer">
    SmartReceipt Inc. © 2026 | 
    <a href="#" style="color: #6b7280; text-decoration: none;">Privacidad</a> | 
    <a href="#" style="color: #6b7280; text-decoration: none;">Términos</a>
</div>
""", unsafe_allow_html=True)