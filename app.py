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
# 1. CONFIGURACIÓN Y ESTILOS (CSS) 🎨
# =======================================================
st.set_page_config(page_title="SmartReceipt Enterprise", layout="wide", page_icon="💳")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    
    .main-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        padding-bottom: 10px;
    }
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
    .stButton > button {
        border-radius: 8px; 
        font-weight: 600;
    }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%; 
        background-color: #ffffff; border-top: 1px solid #e5e7eb; 
        color: #6b7280; text-align: center; padding: 10px; font-size: 12px; z-index: 999;
    }
    [data-testid="stSidebar"] {background-color: #F3F4F6; border-right: 1px solid #E5E7EB;}
    </style>
""", unsafe_allow_html=True)

# TEXTOS LEGALES (CORREGIDO: AHORA SÍ APARECEN)
TERMINOS_CONDICIONES = """
**TÉRMINOS Y CONDICIONES DE USO - SMARTRECEIPT**
1. **ACEPTACIÓN:** Al acceder, acepta estos términos.
2. **NATURALEZA:** Herramienta de IA sujeta a verificación humana.
3. **LICENCIA DE DATOS:** Usted conserva la propiedad de sus tickets. Otorga a SmartReceipt licencia para usar datos de forma **anónima y agregada** para análisis de mercado.
4. **PRIVACIDAD:** Sus datos de contacto son confidenciales.
5. **RESPONSABILIDAD:** No nos hacemos responsables por errores contables derivados del uso de la herramienta.
"""

# =======================================================
# 2. LOGIN Y SEGURIDAD 🔐
# =======================================================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""

def login():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏢 SmartReceipt <span style='font-size: 0.5em; color: #64748B;'>Enterprise</span></h1>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("### Iniciar Sesión")
            with st.form("login_form"):
                usuario = st.text_input("Usuario", placeholder="ej. admin")
                contra = st.text_input("Contraseña", type="password")
                
                st.markdown("---")
                # AQUÍ ESTÁN LOS T&C RECUPERADOS
                with st.expander("📄 Ver Términos y Condiciones"):
                    st.markdown(TERMINOS_CONDICIONES)
                
                acepto = st.checkbox("He leído y acepto los Términos.")
                
                if st.form_submit_button("Acceder", type="primary", use_container_width=True):
                    if not acepto: st.warning("⚠️ Debe aceptar los términos.")
                    elif "usuarios" in st.secrets and usuario in st.secrets["usuarios"]:
                        if st.secrets["usuarios"][usuario] == contra:
                            st.session_state.logged_in = True
                            st.session_state.username = usuario
                            st.rerun()
                        else: st.error("Credenciales inválidas")
                    else: st.error("Usuario no encontrado")

if not st.session_state.logged_in:
    login()
    st.stop()

# =======================================================
# 3. BACKEND 🧠
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

# Inicialización segura
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
# 4. CORE IA & PROCESAMIENTO
# =======================================================
def procesar_imagen_opencv(imagen_pil):
    img_np = np.array(imagen_pil)
    if img_np.shape[-1] == 4: img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else: img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
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
# 5. DASHBOARD & FILTROS AVANZADOS 📊
# =======================================================

# Preparación de datos base
df_local = pd.DataFrame(st.session_state['gastos'])
df_filtrado = pd.DataFrame()

# Limpieza y conversión de tipos para el filtrado
if not df_local.empty:
    for c in ['lat','lon','Monto']:
        if c in df_local.columns: df_local[c] = pd.to_numeric(df_local[c], errors='coerce').fillna(0.0)
    
    # Crear columna auxiliar de MES para el filtro
    df_local['Fecha_dt'] = pd.to_datetime(df_local['Fecha'], dayfirst=True, errors='coerce')
    df_local['Mes_Año'] = df_local['Fecha_dt'].dt.strftime('%Y-%m') # Formato AAAA-MM

with st.sidebar:
    st.markdown(f"""
    <div style="background-color: #E0E7FF; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0; color: #1E3A8A;">👤 {st.session_state.username}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Sincronizar", use_container_width=True):
        st.cache_data.clear()
        if 'gastos' in st.session_state: del st.session_state['gastos'] 
        st.rerun()
    
    st.divider()
    st.header("🌪️ Filtros Avanzados")
    
    # --- MOTOR DE FILTROS 4.0 ---
    if not df_local.empty:
        # 1. Filtro Mes
        opts_mes = sorted([x for x in df_local['Mes_Año'].unique() if x is not None and str(x) != 'nan'], reverse=True)
        sel_mes = st.multiselect("📅 Mes", opts_mes)
        
        # 2. Filtro Categoría
        opts_cat = sorted([str(x) for x in df_local['Categoría'].unique() if x])
        sel_cat = st.multiselect("🏷️ Categoría", opts_cat)
        
        # 3. Filtro Comercio (Nuevo)
        opts_com = sorted([str(x) for x in df_local['Comercio'].unique() if x])
        sel_com = st.multiselect("🏪 Comercio", opts_com)
        
        # 4. Filtro Ubicación/Zona (Nuevo)
        opts_ubi = sorted([str(x) for x in df_local['Ubicación'].unique() if x])
        sel_ubi = st.multiselect("📍 Zona", opts_ubi)
        
        # APLICACIÓN DE FILTROS (Lógica AND)
        df_filtrado = df_local.copy()
        if sel_mes: df_filtrado = df_filtrado[df_filtrado['Mes_Año'].isin(sel_mes)]
        if sel_cat: df_filtrado = df_filtrado[df_filtrado['Categoría'].isin(sel_cat)]
        if sel_com: df_filtrado = df_filtrado[df_filtrado['Comercio'].isin(sel_com)]
        if sel_ubi: df_filtrado = df_filtrado[df_filtrado['Ubicación'].isin(sel_ubi)]
        
        st.caption(f"Mostrando {len(df_filtrado)} de {len(df_local)} tickets")

    st.markdown("---")
    if st.button("Cerrar Sesión", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# --- MAIN LAYOUT ---
st.markdown('<h1 class="main-header">SmartReceipt <span style="font-weight:300;">Analytics</span></h1>', unsafe_allow_html=True)

if not df_filtrado.empty:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><h3>Total</h3><h2 style="color:#1E3A8A">${df_filtrado["Monto"].sum():,.0f}</h2></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h3>Tickets</h3><h2 style="color:#059669">{len(df_filtrado)}</h2></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h3>Promedio</h3><h2 style="color:#D97706">${df_filtrado["Monto"].mean():,.0f}</h2></div>', unsafe_allow_html=True)
    with m4:
        # Categoría Top
        top_cat = df_filtrado.groupby('Categoría')['Monto'].sum().idxmax() if not df_filtrado.empty else "N/A"
        st.markdown(f'<div class="metric-card"><h3>Top Gasto</h3><h3 style="color:#DC2626">{top_cat}</h3></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# TABS
tab_nuevo, tab_dashboard, tab_chat = st.tabs(["📸 Nuevo Ticket", "📈 BI Dashboard", "💬 AI Advisor"])

with tab_nuevo:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### Carga de Comprobante")
        archivo = st.file_uploader("Subir imagen", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, caption="Procesada", use_container_width=True)
            if st.button("⚡ Extraer Datos", type="primary"):
                with st.spinner("AI procesando..."):
                    txt, mod = analizar_ticket(img_proc)
                    if "Error" in txt: st.error(txt)
                    else:
                        try:
                            match = re.search(r'\{.*\}', txt, re.DOTALL)
                            if match:
                                st.session_state['temp_data'] = json.loads(match.group())
                                st.toast("Datos listos", icon="✅")
                        except: st.error("Error formato")

    with col2:
        st.markdown("#### Validación")
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.container(border=True):
                c1,c2 = st.columns(2)
                vc = c1.text_input("Comercio", data.get("comercio",""))
                vm = c2.number_input("Total", value=float(str(data.get("total",0)).replace("$","").replace(",","")))
                c3,c4 = st.columns(2)
                vf = c3.text_input("Fecha", data.get("fecha",""))
                cat_def = data.get("categoria","Varios")
                idx = LISTA_CATEGORIAS.index(cat_def) if cat_def in LISTA_CATEGORIAS else 19
                vcat = c4.selectbox("Categoría", LISTA_CATEGORIAS, index=idx)
                
                with st.expander("📍 Detalles"):
                    vu = st.text_input("Ubicación", data.get("ubicacion",""))
                    vdet = st.text_input("Notas", data.get("detalles",""))
                    vlat = float(data.get("latitud", 0.0))
                    vlon = float(data.get("longitud", 0.0))

                if st.button("💾 Registrar Gasto", type="primary", use_container_width=True):
                    nuevo = {"Usuario": st.session_state.username, "Fecha": vf, "Comercio": vc, "Monto": vm, "Ubicación": vu, "lat": vlat, "lon": vlon, "Categoría": vcat, "Detalles": vdet}
                    st.session_state['gastos'].append(nuevo)
                    hoja = get_google_sheet()
                    if hoja:
                        try: hoja.append_row(list(nuevo.values()))
                        except: pass
                    del st.session_state['temp_data']
                    st.rerun()

with tab_dashboard:
    if not df_filtrado.empty:
        # Gráfica de Barras por Comercio (Nuevo valor agregado)
        st.markdown("##### 🏪 Top Comercios")
        chart_com = alt.Chart(df_filtrado).mark_bar().encode(
            x=alt.X('Monto', title='Gasto Total'),
            y=alt.Y('Comercio', sort='-x'),
            color=alt.Color('Monto', scale={'scheme': 'blues'}),
            tooltip=['Comercio', 'Monto']
        ).properties(height=300)
        st.altair_chart(chart_com, use_container_width=True)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### Distribución")
            st.altair_chart(alt.Chart(df_filtrado).mark_arc(innerRadius=60).encode(theta='Monto', color=alt.Color('Categoría', scale={'scheme': 'tableau10'}), tooltip=['Categoría','Monto']), use_container_width=True)
        with col_g2:
            st.markdown("##### Tendencia Temporal")
            # Gráfica de Línea de tiempo si hay fechas válidas
            if 'Fecha_dt' in df_filtrado.columns:
                line = alt.Chart(df_filtrado).mark_line(point=True).encode(
                    x='Fecha_dt', y='Monto', tooltip=['Fecha', 'Monto', 'Comercio']
                )
                st.altair_chart(line, use_container_width=True)

        map_data = df_filtrado[(df_filtrado['lat']!=0)]
        if not map_data.empty:
            st.markdown("##### 🗺️ Mapa de Calor")
            st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=11),
                layers=[pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_color='[37, 99, 235, 180]', get_radius=200, pickable=True)],
                tooltip={"html": "<b>{Comercio}</b><br/>${Monto}"}))
                
        with st.expander("📂 Ver Datos Crudos"):
            st.dataframe(df_filtrado, use_container_width=True)
    else: st.info("No hay datos que coincidan con los filtros.")

with tab_chat:
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("Pregunta a la IA..."):
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role":"user", "content":q})
        if df_filtrado.empty: r = "Sin datos."
        else:
            with st.spinner("Analizando..."): r = consultar_chat_financiero(q, df_filtrado)
        with st.chat_message("assistant"): st.markdown(r)
        st.session_state['chat_history'].append({"role":"assistant", "content":r})

# FOOTER
st.markdown("""
<div class="footer">
    SmartReceipt Inc. © 2026 | Enterprise Edition
</div>
""", unsafe_allow_html=True)