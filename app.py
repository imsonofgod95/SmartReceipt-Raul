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
# 1. TEXTOS LEGALES ROBUSTOS (COMPLIANCE ARCO) ⚖️
# =======================================================
AVISO_PRIVACIDAD = """
**AVISO DE PRIVACIDAD INTEGRAL**

**1. RESPONSABLE:** SmartReceipt Inc., con domicilio digital en la nube, es responsable del tratamiento de sus datos personales.

**2. DATOS RECABADOS:** Para la prestación del servicio, recabamos: Datos de identificación (Usuario), Datos patrimoniales (Tickets de consumo, montos, comercios) y Datos de geolocalización.

**3. FINALIDADES:**
* **Primarias:** Gestión de gastos, visualización de dashboard y almacenamiento de historial.
* **Secundarias (Negocio):** Análisis estadístico de mercado mediante datos **disociados y anónimos** (Big Data).

**4. DERECHOS ARCO:** Usted tiene derecho a Acceder, Rectificar, Cancelar u Oponerse al tratamiento de sus datos.
* Para ejercer estos derechos, envíe una solicitud a: **legal@smartreceipt.app**
* Su solicitud será atendida en un plazo máximo de 20 días hábiles.

**5. TRANSFERENCIAS:** Sus datos financieros personales NO se venden a terceros. Solo se comercializan insights estadísticos agregados que no permiten su identificación personal.
"""

TERMINOS_CONDICIONES = """
**TÉRMINOS Y CONDICIONES DE USO (SaaS)**

1. **LICENCIA DE USO:** Se otorga una licencia no exclusiva e intransferible para usar el software.
2. **EXENCIÓN DE RESPONSABILIDAD (IA):** El usuario reconoce que el procesamiento es realizado por Inteligencia Artificial (Google Gemini). SmartReceipt no garantiza una precisión del 100% y **se deslinda de responsabilidad por errores contables o fiscales** derivados de la falta de verificación humana por parte del usuario.
3. **PROPIEDAD INTELECTUAL:** El software es propiedad de SmartReceipt. Los tickets son propiedad del usuario.
4. **CONSENTIMIENTO DE DATOS:** Al usar la app, el usuario autoriza el uso de su información de consumo para la generación de reportes macroeconómicos anónimos.
"""

# =======================================================
# 2. CONFIGURACIÓN Y ESTILOS UI 🎨
# =======================================================
st.set_page_config(page_title="SmartReceipt Enterprise", layout="wide", page_icon="⚖️")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    
    .main-header {
        background: linear-gradient(90deg, #0F172A 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        padding-bottom: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {font-size: 2rem; font-weight: 700; color: #0F172A;}
    .metric-label {font-size: 0.875rem; color: #64748B; font-weight: 500;}
    
    /* Estilo para los Highlights */
    .highlight-box {
        background-color: #F0F9FF;
        border-left: 5px solid #0284C7;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    .stButton > button {
        border-radius: 8px; font-weight: 600; border: none;
        transition: all 0.2s;
    }
    .stButton > button:hover {transform: translateY(-2px);}
    
    [data-testid="stSidebar"] {background-color: #F8FAFC; border-right: 1px solid #E2E8F0;}
    </style>
""", unsafe_allow_html=True)

# =======================================================
# 3. LOGIN BLINDADO 🔐
# =======================================================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""

def login():
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #0F172A;'>🏢 SmartReceipt</h1>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("### Acceso Seguro")
            with st.form("login_form"):
                usuario = st.text_input("Usuario Corporativo")
                contra = st.text_input("Contraseña", type="password")
                
                st.markdown("---")
                st.caption("Documentación Legal Obligatoria:")
                with st.expander("📜 Leer Aviso de Privacidad (ARCO)"):
                    st.markdown(AVISO_PRIVACIDAD)
                with st.expander("⚖️ Leer Términos y Condiciones"):
                    st.markdown(TERMINOS_CONDICIONES)
                
                check_privacidad = st.checkbox("He leído el Aviso de Privacidad.")
                check_terminos = st.checkbox("Acepto los Términos y Condiciones.")
                
                if st.form_submit_button("Ingresar al Sistema", type="primary", use_container_width=True):
                    if not (check_privacidad and check_terminos):
                        st.error("🛑 Para cumplir con la normativa legal, debe aceptar ambos documentos.")
                    elif "usuarios" in st.secrets and usuario in st.secrets["usuarios"]:
                        if st.secrets["usuarios"][usuario] == contra:
                            st.session_state.logged_in = True
                            st.session_state.username = usuario
                            st.rerun()
                        else: st.error("Credenciales inválidas")
                    else: st.error("Usuario no registrado")

if not st.session_state.logged_in:
    login()
    st.stop()

# =======================================================
# 4. BACKEND & CONEXIONES 🧠
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

# Inicialización
if 'gastos' not in st.session_state or not st.session_state['gastos']:
    hoja = get_google_sheet()
    if hoja:
        try:
            if hoja.acell('A1').value != "Usuario":
                hoja.insert_row(["Usuario", "Fecha", "Hora", "Comercio", "Monto", "Ubicación", "lat", "lon", "Categoría", "Detalles"], 1)
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
    "Mantenimiento Automotriz", "Impuestos y Predial", "Varios"
]

# =======================================================
# 5. FUNCIONES CORE (IA & VISIÓN)
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
        
        # --- PROMPT V28: MEJORADO PARA MANUSCRITOS ---
        prompt = f"""
        Analiza la imagen. Puede ser:
        1. TICKET DE COMPRA (Impreso)
        2. RECIBO DE SERVICIOS (Luz, Agua, Gas, Predial)
        3. NOTA DE REMISIÓN (MANUSCRITA / A MANO ALZADA)
        
        INSTRUCCIONES CLAVE:
        - Si es MANUSCRITO: Haz tu mejor esfuerzo OCR. Si la fecha dice "Noviembre", conviértelo a formato numérico (DD/MM/AAAA).
        - Comercio: Si es nota de remisión genérica, busca el nombre del negocio o la persona que firma.
        - Total: Busca el "Total", "Neto" o la cifra final.
        - Fecha: Formato DD/MM/AAAA obligatorio.
        
        CATEGORÍAS DISPONIBLES: [{cats_str}]
        
        JSON OBLIGATORIO: {{"comercio": "Nombre", "total": 0.00, "fecha": "DD/MM/AAAA", "hora": "HH:MM", "ubicacion": "Sucursal o Ciudad", "latitud": 0.0, "longitud": 0.0, "categoria": "Texto", "detalles": "Texto"}}
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

def safe_float(val):
    try:
        if val is None: return 0.0
        return float(val)
    except: return 0.0

# =======================================================
# 6. DASHBOARD & SIDEBAR PROFESIONAL
# =======================================================
df_local = pd.DataFrame(st.session_state['gastos'])
df_filtrado = pd.DataFrame()

# Preprocesamiento de datos GLOBAL
if not df_local.empty:
    for c in ['lat','lon','Monto']:
        if c in df_local.columns: df_local[c] = pd.to_numeric(df_local[c], errors='coerce').fillna(0.0)
    df_local['Fecha_dt'] = pd.to_datetime(df_local['Fecha'], dayfirst=True, errors='coerce')
    df_local['Mes_Año'] = df_local['Fecha_dt'].dt.strftime('%Y-%m')

with st.sidebar:
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 15px; border: 1px solid #e2e8f0; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0; color: #0F172A;">👤 {st.session_state.username}</h3>
        <p style="color: green; font-size: 10px;">● Online</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Sincronizar Datos", use_container_width=True):
        st.cache_data.clear()
        if 'gastos' in st.session_state: del st.session_state['gastos'] 
        st.rerun()
    
    st.divider()
    st.markdown("### 🎯 Filtros de Datos")
    
    if not df_local.empty:
        opts_mes = sorted([x for x in df_local['Mes_Año'].unique() if x is not None and str(x) != 'nan'], reverse=True)
        sel_mes = st.multiselect("📅 Periodo", opts_mes)
        opts_cat = sorted([str(x) for x in df_local['Categoría'].unique() if x])
        sel_cat = st.multiselect("🏷️ Categoría", opts_cat)
        opts_com = sorted([str(x) for x in df_local['Comercio'].unique() if x])
        sel_com = st.multiselect("🏪 Comercio", opts_com)
        
        df_filtrado = df_local.copy()
        if sel_mes: df_filtrado = df_filtrado[df_filtrado['Mes_Año'].isin(sel_mes)]
        if sel_cat: df_filtrado = df_filtrado[df_filtrado['Categoría'].isin(sel_cat)]
        if sel_com: df_filtrado = df_filtrado[df_filtrado['Comercio'].isin(sel_com)]
    
    st.divider()
    with st.expander("🛡️ Derechos ARCO"):
        st.caption("Para ejercer sus derechos de Acceso, Rectificación, Cancelación u Oposición, contacte a:")
        st.markdown("**legal@smartreceipt.app**")
        st.caption("Referencia: Compliance LFPDPPP")
        
    if st.button("Cerrar Sesión", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# --- MAIN CONTENT ---
st.markdown('<h1 class="main-header">SmartReceipt <span style="font-weight:300;">Enterprise</span></h1>', unsafe_allow_html=True)

if not df_filtrado.empty:
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">Gasto Total</div><div class="metric-value" style="color:#0F172A">${df_filtrado["Monto"].sum():,.0f}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">Transacciones</div><div class="metric-value" style="color:#3B82F6">{len(df_filtrado)}</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">Ticket Promedio</div><div class="metric-value" style="color:#F59E0B">${df_filtrado["Monto"].mean():,.0f}</div></div>', unsafe_allow_html=True)
    with m4:
        top_cat = df_filtrado.groupby('Categoría')['Monto'].sum().idxmax() if not df_filtrado.empty else "-"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Mayor Gasto</div><div class="metric-value" style="color:#EF4444; font-size:1.5rem">{top_cat}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# TABS
tab_nuevo, tab_dashboard, tab_chat = st.tabs(["📸 Nuevo Ticket", "📈 Dashboard BI", "💬 AI Assistant"])

with tab_nuevo:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### 1. Digitalización (Universal)")
        archivo = st.file_uploader("Subir comprobante", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, caption="Procesada para OCR", use_container_width=True)
            if st.button("⚡ Analizar con Gemini AI", type="primary"):
                with st.spinner("Extrayendo metadatos..."):
                    txt, mod = analizar_ticket(img_proc)
                    if "Error" in txt: st.error(txt)
                    else:
                        try:
                            match = re.search(r'\{.*\}', txt, re.DOTALL)
                            if match:
                                st.session_state['temp_data'] = json.loads(match.group())
                                st.toast("Lectura completada", icon="✅")
                        except: st.error("Error de formato")

    with col2:
        st.markdown("#### 2. Validación Fiscal")
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.container(border=True):
                c1,c2 = st.columns(2)
                vc = c1.text_input("Comercio / Proveedor", data.get("comercio",""))
                try:
                    val_monto = float(str(data.get("total",0)).replace("$","").replace(",",""))
                except: val_monto = 0.0
                vm = c2.number_input("Monto Total ($)", value=val_monto)

                c3,c4,c5 = st.columns(3)
                vf = c3.text_input("Fecha", data.get("fecha",""))
                vh = c4.text_input("Hora", data.get("hora", "00:00"))
                
                cat_def = data.get("categoria","Varios")
                if "Servicios" in cat_def or "Luz" in cat_def or "Agua" in cat_def or "CFE" in cat_def or "Gas" in cat_def:
                     idx = LISTA_CATEGORIAS.index("Servicios (Luz/Agua/Internet)")
                elif "Predial" in cat_def or "Impuesto" in cat_def:
                     idx = LISTA_CATEGORIAS.index("Impuestos y Predial")
                elif cat_def in LISTA_CATEGORIAS:
                     idx = LISTA_CATEGORIAS.index(cat_def)
                else:
                     idx = LISTA_CATEGORIAS.index("Varios")

                vcat = c5.selectbox("Categoría", LISTA_CATEGORIAS, index=idx)
                
                with st.expander("📍 Geolocalización y Notas"):
                    vu = st.text_input("Sucursal", data.get("ubicacion",""))
                    vdet = st.text_input("Concepto / Periodo", data.get("detalles",""))
                    vlat = safe_float(data.get("latitud"))
                    vlon = safe_float(data.get("longitud"))

                if st.button("💾 Guardar Transacción", type="primary", use_container_width=True):
                    nuevo = {"Usuario": st.session_state.username, "Fecha": vf, "Hora": vh, "Comercio": vc, "Monto": vm, "Ubicación": vu, "lat": vlat, "lon": vlon, "Categoría": vcat, "Detalles": vdet}
                    st.session_state['gastos'].append(nuevo)
                    hoja = get_google_sheet()
                    if hoja:
                        try: hoja.append_row(list(nuevo.values()))
                        except: pass
                    del st.session_state['temp_data']
                    st.rerun()

with tab_dashboard:
    if not df_filtrado.empty:
        st.markdown("### 💡 Highlights del Periodo")
        hc1, hc2, hc3 = st.columns(3)
        
        idx_max = df_filtrado['Monto'].idxmax()
        row_max = df_filtrado.loc[idx_max]
        with hc1:
            st.info(f"💸 **Compra más grande:**\n\n${row_max['Monto']:,.2f} en **{row_max['Comercio']}** ({row_max['Fecha']}).")
        
        cat_top = df_filtrado.groupby('Categoría')['Monto'].sum().idxmax()
        monto_cat = df_filtrado.groupby('Categoría')['Monto'].sum().max()
        with hc2:
            st.success(f"🛍️ **Categoría Top:**\n\n**{cat_top}** (${monto_cat:,.2f}).")
            
        servicios = df_filtrado[df_filtrado['Categoría'].str.contains("Servicios", case=False, na=False)]
        total_serv = servicios['Monto'].sum() if not servicios.empty else 0
        with hc3:
            st.warning(f"⚡ **Gastos en Servicios:**\n\nTotal: **${total_serv:,.2f}** (Luz, Agua, etc.)")
        
        st.markdown("---")
        
        st.markdown("##### 🏢 Gasto por Comercio")
        chart_bar = alt.Chart(df_filtrado).mark_bar(cornerRadius=5).encode(
            x=alt.X('Monto', title='Monto Total'),
            y=alt.Y('Comercio', sort='-x'),
            color=alt.Color('Monto', scale={'scheme': 'blues'}),
            tooltip=['Comercio', 'Monto', 'Fecha']
        ).properties(height=300)
        st.altair_chart(chart_bar, use_container_width=True)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### Distribución")
            base = alt.Chart(df_filtrado).encode(theta=alt.Theta("Monto", stack=True))
            pie = base.mark_arc(innerRadius=60).encode(
                color=alt.Color("Categoría", scale={'scheme': 'tableau10'}),
                tooltip=["Categoría", "Monto"]
            )
            st.altair_chart(pie, use_container_width=True)
        with col_g2:
            st.markdown("##### Historial Temporal")
            if 'Fecha_dt' in df_filtrado.columns:
                line = alt.Chart(df_filtrado).mark_line(point=True, interpolate='monotone').encode(
                    x='Fecha_dt', y='Monto', tooltip=['Fecha', 'Monto', 'Comercio']
                )
                st.altair_chart(line, use_container_width=True)

        map_data = df_filtrado[(df_filtrado['lat']!=0)]
        if not map_data.empty:
            st.markdown("##### 🗺️ Mapa de Operaciones")
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=11),
                layers=[pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position='[lon, lat]',
                    get_color=[15, 23, 42, 200],
                    get_radius=200,
                    pickable=True
                )],
                tooltip={"html": "<b>{Comercio}</b><br/>${Monto}"}
            ))
                
        with st.expander("📂 Exportar Datos"):
            st.dataframe(df_filtrado, use_container_width=True)
    else: st.info("No hay datos disponibles para los filtros seleccionados.")

with tab_chat:
    st.caption("Asistente financiero potenciado por Gemini 1.5. Pregunta sobre patrones de gasto.")
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input("Ej: ¿Cuál fue mi gasto más alto en Restaurantes?"):
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role":"user", "content":q})
        if df_filtrado.empty: r = "Sin datos."
        else:
            with st.spinner("Analizando..."): r = consultar_chat_financiero(q, df_filtrado)
        with st.chat_message("assistant"): st.markdown(r)
        st.session_state['chat_history'].append({"role":"assistant", "content":r})

# FOOTER
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #94a3b8; font-size: 12px;">
    SmartReceipt Inc. © 2026 | Cumplimiento LFPDPPP | 
    <a href="#" style="color: #64748b;">Privacidad</a>
</div>
""", unsafe_allow_html=True)