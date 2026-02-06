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
# 1. CEREBRO BILINGÜE (DICCIONARIOS DE TRADUCCIÓN) 🌍
# =======================================================
TEXTOS = {
    "ES": {
        "login_title": "Acceso Seguro",
        "user_label": "Usuario Corporativo",
        "pass_label": "Contraseña",
        "login_btn": "Ingresar al Sistema",
        "legal_check1": "He leído el Aviso de Privacidad",
        "legal_check2": "Acepto los Términos y Condiciones",
        "legal_error": "🛑 Debe aceptar los términos legales.",
        "welcome": "Bienvenido",
        "sync_btn": "🔄 Sincronizar Datos",
        "filters": "🎯 Filtros de Datos",
        "period": "📅 Periodo",
        "category": "🏷️ Categoría",
        "commerce": "🏪 Comercio",
        "logout": "Cerrar Sesión",
        "tab1": "📸 Nuevo Ticket",
        "tab2": "📈 Dashboard BI",
        "tab3": "💬 AI Assistant",
        "upload_label": "Subir comprobante",
        "manual_btn": "✍️ Captura Manual (Sin recibo)",
        "manual_info": "¿Gasto sin comprobante? (Propinas, Taxis, etc.)",
        "analyze_btn": "⚡ Analizar con Gemini AI",
        "save_btn": "💾 Guardar Transacción",
        "highlights_title": "💡 Highlights del Periodo",
        "highlight_max": "💸 Compra más grande",
        "highlight_top": "🛍️ Categoría Top",
        "highlight_serv": "⚡ Gastos en Servicios",
        "total_label": "Gasto Total",
        "trans_label": "Transacciones",
        "avg_label": "Ticket Promedio",
        "max_label": "Mayor Gasto",
        "delete_title": "🗑️ Gestión de Registros",
        "delete_caption": "Selecciona un registro para eliminarlo permanentemente.",
        "delete_select": "Seleccionar Gasto a Eliminar",
        "delete_btn": "Eliminar Registro",
        "delete_success": "Registro eliminado de la Nube",
        "chat_placeholder": "Ej: ¿En qué gasté más este mes?",
        "legal_privacy": "**AVISO DE PRIVACIDAD:** Sus datos son usados para gestión de gastos.",
        "legal_terms": "**TÉRMINOS:** Uso bajo su responsabilidad. IA puede cometer errores."
    },
    "EN": {
        "login_title": "Secure Access",
        "user_label": "Corporate User",
        "pass_label": "Password",
        "login_btn": "Login to System",
        "legal_check1": "I have read the Privacy Policy",
        "legal_check2": "I accept Terms & Conditions",
        "legal_error": "🛑 You must accept legal terms.",
        "welcome": "Welcome",
        "sync_btn": "🔄 Sync Data",
        "filters": "🎯 Data Filters",
        "period": "📅 Period",
        "category": "🏷️ Category",
        "commerce": "🏪 Merchant",
        "logout": "Logout",
        "tab1": "📸 New Receipt",
        "tab2": "📈 BI Dashboard",
        "tab3": "💬 AI Assistant",
        "upload_label": "Upload receipt",
        "manual_btn": "✍️ Manual Entry (No receipt)",
        "manual_info": "Expense without receipt? (Tips, Cabs, etc.)",
        "analyze_btn": "⚡ Analyze with Gemini AI",
        "save_btn": "💾 Save Transaction",
        "highlights_title": "💡 Period Highlights",
        "highlight_max": "💸 Biggest Purchase",
        "highlight_top": "🛍️ Top Category",
        "highlight_serv": "⚡ Utilities Expenses",
        "total_label": "Total Spend",
        "trans_label": "Transactions",
        "avg_label": "Avg Ticket",
        "max_label": "Top Expense",
        "delete_title": "🗑️ Record Management",
        "delete_caption": "Select a record to delete permanently.",
        "delete_select": "Select Expense to Delete",
        "delete_btn": "Delete Record",
        "delete_success": "Record deleted from Cloud",
        "chat_placeholder": "Ex: What was my highest expense?",
        "legal_privacy": "**PRIVACY POLICY:** Data used for expense management.",
        "legal_terms": "**TERMS:** Use at your own risk. AI might make mistakes."
    }
}

CATEGORIAS = {
    "ES": [
        "Alimentos y Supermercado", "Restaurantes y Bares", "Gasolina y Transporte",
        "Salud y Farmacia", "Hogar y Muebles", "Servicios (Luz/Agua/Internet)", 
        "Telefonía", "Ropa y Calzado", "Electrónica", "Entretenimiento", 
        "Educación", "Mascotas", "Regalos", "Viajes", "Suscripciones",
        "Cuidado Personal", "Deportes", "Oficina", "Mantenimiento Auto", 
        "Impuestos y Predial", "Varios"
    ],
    "EN": [
        "Groceries & Supermarket", "Restaurants & Bars", "Gas & Transport",
        "Health & Pharmacy", "Home & Furniture", "Utilities (Water/Electric)", 
        "Phone & Internet", "Clothing", "Electronics", "Entertainment", 
        "Education", "Pets", "Gifts", "Travel", "Subscriptions",
        "Personal Care", "Sports", "Office", "Car Maintenance", 
        "Taxes", "Misc"
    ]
}

# =======================================================
# 2. CONFIGURACIÓN Y ESTILOS UI 🎨
# =======================================================
st.set_page_config(page_title="SmartReceipt Enterprise", layout="wide", page_icon="🌎")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    .main-header {
        background: linear-gradient(90deg, #0F172A 0%, #334155 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; padding-bottom: 10px;
    }
    .metric-card {
        background-color: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 24px; text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {font-size: 2rem; font-weight: 700; color: #0F172A;}
    .metric-label {font-size: 0.875rem; color: #64748B; font-weight: 500;}
    .highlight-box {
        background-color: #F0F9FF; border-left: 5px solid #0284C7;
        padding: 15px; border-radius: 5px; margin-bottom: 10px;
    }
    .stButton > button {border-radius: 8px; font-weight: 600;}
    [data-testid="stSidebar"] {background-color: #F8FAFC; border-right: 1px solid #E2E8F0;}
    </style>
""", unsafe_allow_html=True)

# =======================================================
# 3. GESTIÓN DE IDIOMA Y LOGIN 🔐
# =======================================================
if "language" not in st.session_state: st.session_state.language = "ES"
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""

def login():
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        # Selector de idioma
        lang_sel = st.selectbox("Language / Idioma", ["Español", "English"], key="lang_login")
        st.session_state.language = "ES" if lang_sel == "Español" else "EN"
        t = TEXTOS[st.session_state.language]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #0F172A;'>🏢 SmartReceipt</h1>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown(f"### {t['login_title']}")
            with st.form("login_form"):
                usuario = st.text_input(t['user_label'])
                contra = st.text_input(t['pass_label'], type="password")
                
                st.markdown("---")
                with st.expander("⚖️ Legal"):
                    st.markdown(t['legal_privacy'])
                    st.markdown(t['legal_terms'])
                
                check1 = st.checkbox(t['legal_check1'])
                check2 = st.checkbox(t['legal_check2'])
                
                if st.form_submit_button(t['login_btn'], type="primary", use_container_width=True):
                    if not (check1 and check2):
                        st.error(t['legal_error'])
                    elif "usuarios" in st.secrets and usuario in st.secrets["usuarios"]:
                        if st.secrets["usuarios"][usuario] == contra:
                            st.session_state.logged_in = True
                            st.session_state.username = usuario
                            st.rerun()
                        else: st.error("Error")
                    else: st.error("Error")

if not st.session_state.logged_in:
    login()
    st.stop()

# Cargar textos dinámicos
T = TEXTOS[st.session_state.language]
CATS_ACTUALES = CATEGORIAS[st.session_state.language]

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

def analizar_ticket(imagen_pil, idioma_actual):
    try:
        mods = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        modelo = next((m for m in mods if 'flash' in m and '1.5' in m), mods[0] if mods else "gemini-1.5-flash")
    except: modelo = "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(modelo)
        cats_str = ", ".join(CATS_ACTUALES)
        
        # PROMPT DINÁMICO SEGÚN IDIOMA
        instrucciones_extra = ""
        if idioma_actual == "EN":
            instrucciones_extra = "Output 'categoria' and 'detalles' IN ENGLISH matching the list."
        else:
            instrucciones_extra = "Extrae 'categoria' y 'detalles' EN ESPAÑOL coincidiendo con la lista."

        prompt = f"""
        Analiza imagen (Ticket, Recibo, Manuscrito).
        {instrucciones_extra}
        
        LISTA DE CATEGORÍAS VÁLIDAS: [{cats_str}]
        
        JSON OBLIGATORIO: {{"comercio": "Nombre", "total": 0.00, "fecha": "DD/MM/AAAA", "hora": "HH:MM", "ubicacion": "Sucursal", "latitud": 0.0, "longitud": 0.0, "categoria": "Texto", "detalles": "Texto"}}
        """
        response = model.generate_content([prompt, imagen_pil])
        return response.text, modelo
    except Exception as e: return f"Error: {e}", modelo

def consultar_chat_financiero(pregunta, datos_df, idioma_actual):
    try:
        mods = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        modelo = next((m for m in mods if 'flash' in m and '1.5' in m), mods[0] if mods else "gemini-1.5-flash")
    except: modelo = "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(modelo)
        datos_csv = datos_df.to_csv(index=False)
        lang_prompt = "Answer in ENGLISH" if idioma_actual == "EN" else "Responde en ESPAÑOL"
        prompt = f"Role: Financial Assistant. {lang_prompt}. Data: \n---\n{datos_csv}\n---\nQuery: {pregunta}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error Chat: {e}"

def safe_float(val):
    try:
        if val is None: return 0.0
        return float(val)
    except: return 0.0

# =======================================================
# 6. DASHBOARD & SIDEBAR
# =======================================================
df_local = pd.DataFrame(st.session_state['gastos'])
df_filtrado = pd.DataFrame()

if not df_local.empty:
    for c in ['lat','lon','Monto']:
        if c in df_local.columns: df_local[c] = pd.to_numeric(df_local[c], errors='coerce').fillna(0.0)
    df_local['Fecha_dt'] = pd.to_datetime(df_local['Fecha'], dayfirst=True, errors='coerce')
    df_local['Mes_Año'] = df_local['Fecha_dt'].dt.strftime('%Y-%m')

with st.sidebar:
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 15px; border: 1px solid #e2e8f0; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0; color: #0F172A;">👤 {st.session_state.username}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Selector en Sidebar
    lang_side = st.selectbox("🌐 Language", ["Español", "English"], index=0 if st.session_state.language=="ES" else 1)
    if (lang_side == "Español" and st.session_state.language != "ES") or (lang_side == "English" and st.session_state.language != "EN"):
        st.session_state.language = "ES" if lang_side == "Español" else "EN"
        st.rerun()

    if st.button(T['sync_btn'], use_container_width=True):
        st.cache_data.clear()
        if 'gastos' in st.session_state: del st.session_state['gastos'] 
        st.rerun()
    
    st.divider()
    st.markdown(f"### {T['filters']}")
    
    if not df_local.empty:
        opts_mes = sorted([x for x in df_local['Mes_Año'].unique() if str(x) != 'nan'], reverse=True)
        sel_mes = st.multiselect(T['period'], opts_mes)
        opts_cat = sorted([str(x) for x in df_local['Categoría'].unique() if x])
        sel_cat = st.multiselect(T['category'], opts_cat)
        opts_com = sorted([str(x) for x in df_local['Comercio'].unique() if x])
        sel_com = st.multiselect(T['commerce'], opts_com)
        
        df_filtrado = df_local.copy()
        if sel_mes: df_filtrado = df_filtrado[df_filtrado['Mes_Año'].isin(sel_mes)]
        if sel_cat: df_filtrado = df_filtrado[df_filtrado['Categoría'].isin(sel_cat)]
        if sel_com: df_filtrado = df_filtrado[df_filtrado['Comercio'].isin(sel_com)]
    
    st.divider()
    if st.button(T['logout'], use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# --- MAIN CONTENT ---
st.markdown('<h1 class="main-header">SmartReceipt <span style="font-weight:300;">Global</span></h1>', unsafe_allow_html=True)

if not df_filtrado.empty:
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">{T["total_label"]}</div><div class="metric-value" style="color:#0F172A">${df_filtrado["Monto"].sum():,.0f}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">{T["trans_label"]}</div><div class="metric-value" style="color:#3B82F6">{len(df_filtrado)}</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">{T["avg_label"]}</div><div class="metric-value" style="color:#F59E0B">${df_filtrado["Monto"].mean():,.0f}</div></div>', unsafe_allow_html=True)
    with m4:
        top_cat = df_filtrado.groupby('Categoría')['Monto'].sum().idxmax() if not df_filtrado.empty else "-"
        st.markdown(f'<div class="metric-card"><div class="metric-label">{T["max_label"]}</div><div class="metric-value" style="color:#EF4444; font-size:1.5rem">{top_cat}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

tab_nuevo, tab_dashboard, tab_chat = st.tabs([T['tab1'], T['tab2'], T['tab3']])

with tab_nuevo:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### 1. Digitalization")
        archivo = st.file_uploader(T['upload_label'], type=["jpg","png","jpeg"], label_visibility="collapsed")
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, caption="OCR Ready", use_container_width=True)
            if st.button(T['analyze_btn'], type="primary"):
                with st.spinner("AI Working..."):
                    txt, mod = analizar_ticket(img_proc, st.session_state.language) # Pasa idioma
                    if "Error" in txt: st.error(txt)
                    else:
                        try:
                            match = re.search(r'\{.*\}', txt, re.DOTALL)
                            if match:
                                st.session_state['temp_data'] = json.loads(match.group())
                                st.rerun()
                        except: st.error("Format Error")
        
        # --- CAPTURA MANUAL BILINGÜE ---
        st.markdown("---")
        st.info(T['manual_info'])
        if st.button(T['manual_btn'], use_container_width=True):
            st.session_state['temp_data'] = {
                "comercio": "", "total": 0.0,
                "fecha": datetime.now().strftime("%d/%m/%Y"),
                "hora": datetime.now().strftime("%H:%M"),
                "categoria": CATS_ACTUALES[0], # Categoría por defecto en idioma actual
                "ubicacion": "", "detalles": "Manual", "latitud": 0.0, "longitud": 0.0
            }
            st.rerun()

    with col2:
        st.markdown("#### 2. Validation")
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.container(border=True):
                c1,c2 = st.columns(2)
                vc = c1.text_input(T['commerce'], data.get("comercio",""))
                try: val_monto = float(str(data.get("total",0)).replace("$","").replace(",",""))
                except: val_monto = 0.0
                vm = c2.number_input("Total ($)", value=val_monto)

                c3,c4,c5 = st.columns(3)
                vf = c3.text_input("Date (DD/MM/YYYY)", data.get("fecha",""))
                vh = c4.text_input("Time", data.get("hora", "00:00"))
                
                # Mapeo de Categorías Inteligente
                cat_def = data.get("categoria","Misc")
                idx = 0
                if cat_def in CATS_ACTUALES:
                    idx = CATS_ACTUALES.index(cat_def)
                else:
                    idx = len(CATS_ACTUALES) - 1 # Varios/Misc
                
                vcat = c5.selectbox(T['category'], CATS_ACTUALES, index=idx)
                
                with st.expander("📍 GPS & Notes"):
                    vu = st.text_input("Location", data.get("ubicacion",""))
                    vdet = st.text_input("Details", data.get("detalles",""))
                    vlat = safe_float(data.get("latitud"))
                    vlon = safe_float(data.get("longitud"))

                if st.button(T['save_btn'], type="primary", use_container_width=True):
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
        # Highlights Bilingües
        st.markdown(f"### {T['highlights_title']}")
        hc1, hc2, hc3 = st.columns(3)
        idx_max = df_filtrado['Monto'].idxmax()
        row_max = df_filtrado.loc[idx_max]
        with hc1: st.info(f"{T['highlight_max']}:\n\n${row_max['Monto']:,.2f} - **{row_max['Comercio']}**")
        
        cat_top = df_filtrado.groupby('Categoría')['Monto'].sum().idxmax()
        monto_cat = df_filtrado.groupby('Categoría')['Monto'].sum().max()
        with hc2: st.success(f"{T['highlight_top']}:\n\n**{cat_top}** (${monto_cat:,.2f}).")
        
        servicios = df_filtrado[df_filtrado['Categoría'].str.contains("Servicios|Utilities", case=False, na=False)]
        total_serv = servicios['Monto'].sum() if not servicios.empty else 0
        with hc3: st.warning(f"{T['highlight_serv']}:\n\nTotal: **${total_serv:,.2f}**")
        
        st.markdown("---")
        # Gráficos
        chart_bar = alt.Chart(df_filtrado).mark_bar(cornerRadius=5).encode(
            x=alt.X('Monto', title='Total'), y=alt.Y('Comercio', sort='-x'),
            color=alt.Color('Monto', scale={'scheme': 'blues'}), tooltip=['Comercio', 'Monto']
        ).properties(height=300)
        st.altair_chart(chart_bar, use_container_width=True)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            base = alt.Chart(df_filtrado).encode(theta=alt.Theta("Monto", stack=True))
            pie = base.mark_arc(innerRadius=60).encode(
                color=alt.Color("Categoría", scale={'scheme': 'tableau10'}), tooltip=["Categoría", "Monto"]
            )
            st.altair_chart(pie, use_container_width=True)
        with col_g2:
            if 'Fecha_dt' in df_filtrado.columns:
                line = alt.Chart(df_filtrado).mark_line(point=True).encode(x='Fecha_dt', y='Monto', tooltip=['Fecha', 'Monto'])
                st.altair_chart(line, use_container_width=True)

        # --- SECCIÓN DE GESTIÓN (BORRAR) CON TRADUCCIÓN ---
        st.markdown(f"### {T['delete_title']}")
        st.caption(T['delete_caption'])
        
        opciones_borrar = {f"{i} | {r['Fecha']} - {r['Comercio']} (${r['Monto']})": i for i, r in df_filtrado.iterrows()}
        c_del1, c_del2 = st.columns([3,1])
        with c_del1: 
            sel_del = st.selectbox(T['delete_select'], list(opciones_borrar.keys()))
        with c_del2: 
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(T['delete_btn'], type="primary"):
                idx_real = opciones_borrar[sel_del]
                hoja = get_google_sheet()
                if hoja:
                    try: 
                        hoja.delete_rows(idx_real + 2)
                        del st.session_state['gastos'][idx_real]
                        st.toast(T['delete_success'], icon="🗑️")
                        st.rerun()
                    except: st.error("Error DB")
        
        with st.expander("📂 Data"): st.dataframe(df_filtrado, use_container_width=True)
    else: st.info("No data / Sin datos.")

with tab_chat:
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input(T['chat_placeholder']):
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role":"user", "content":q})
        if df_filtrado.empty: r = "No data."
        else:
            with st.spinner("AI Thinking..."): 
                r = consultar_chat_financiero(q, df_filtrado, st.session_state.language) # Pasa idioma
        with st.chat_message("assistant"): st.markdown(r)
        st.session_state['chat_history'].append({"role":"assistant", "content":r})

# FOOTER
st.markdown("<div style='text-align: center; margin-top: 50px; color: #94a3b8; font-size: 12px;'>SmartReceipt Global © 2026</div>", unsafe_allow_html=True)