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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import io

# --- LIBRERÍA PARA PDF (IMPORTACIÓN SEGURA) ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# =======================================================
# 1. CEREBRO BILINGÜE (TEXTOS) 🌍
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
        "tab1": "📸 Digitalizar Ticket",
        "tab2": "📈 Dashboard BI",
        "tab3": "💬 AI Assistant",
        "upload_label": "📂 Toca para usar CÁMARA o GALERÍA",
        "manual_btn": "✍️ Captura Manual",
        "manual_info": "¿Gasto sin comprobante?",
        "analyze_btn": "⚡ PROCESAR TICKET",
        "save_btn": "💾 Guardar Transacción",
        "validation_title": "✅ Validar Información Extraída",
        "highlights_title": "💡 Estado Financiero",
        "highlight_max": "💸 Compra más grande",
        "highlight_top": "🛍️ Categoría Top",
        "highlight_serv": "⚡ Servicios Básicos",
        "budget_set": "⚙️ Configurar Presupuestos",   
        "budget_used": "Consumido Global",             
        "total_label": "Gasto Total",
        "trans_label": "Transacciones",
        "avg_label": "Ticket Promedio",
        "max_label": "Mayor Gasto",
        "chart_budget_title": "📊 Control Presupuestal (Gasto vs Límite)", 
        "delete_title": "🗑️ Gestión de Registros",
        "delete_caption": "Selecciona un registro para eliminarlo permanentemente.",
        "delete_select": "Seleccionar Gasto a Eliminar",
        "delete_btn": "Eliminar Registro",
        "delete_success": "Registro eliminado de la Nube",
        "chat_placeholder": "Ej: ¿En qué gasté más este mes?",
        "legal_privacy": "**AVISO DE PRIVACIDAD:** Sus datos son usados para gestión de gastos.",
        "legal_terms": "**TÉRMINOS:** Uso bajo su responsabilidad. IA puede cometer errores.",
        "preview_label": "Vista previa (Lista para IA):",
        "report_title": "📑 Centro de Reportes",
        "download_pdf": "📄 Descargar PDF Ejecutivo",
        "download_csv": "📊 Descargar Excel (CSV)",
        "report_error": "⚠️ Instala 'reportlab' para generar PDFs."
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
        "tab1": "📸 Digitize Receipt",
        "tab2": "📈 BI Dashboard",
        "tab3": "💬 AI Assistant",
        "upload_label": "📂 Tap to use CAMERA or GALLERY",
        "manual_btn": "✍️ Manual Entry",
        "manual_info": "Expense without receipt?",
        "analyze_btn": "⚡ PROCESS RECEIPT",
        "save_btn": "💾 Save Transaction",
        "validation_title": "✅ Validate Extracted Data",
        "highlights_title": "💡 Financial Status",
        "highlight_max": "💸 Biggest Purchase",
        "highlight_top": "🛍️ Top Category",
        "highlight_serv": "⚡ Basic Utilities",
        "budget_set": "⚙️ Set Category Budgets",       
        "budget_used": "Global Used",                  
        "total_label": "Total Spend",
        "trans_label": "Transactions",
        "avg_label": "Avg Ticket",
        "max_label": "Top Expense",
        "chart_budget_title": "📊 Budget Control (Spend vs Limit)", 
        "delete_title": "🗑️ Record Management",
        "delete_caption": "Select a record to delete permanently.",
        "delete_select": "Select Expense to Delete",
        "delete_btn": "Delete Record",
        "delete_success": "Record deleted from Cloud",
        "chat_placeholder": "Ex: What was my highest expense?",
        "legal_privacy": "**PRIVACY POLICY:** Data used for expense management.",
        "legal_terms": "**TERMS:** Use at your own risk. AI might make mistakes.",
        "preview_label": "Preview (AI Ready):",
        "report_title": "📑 Report Center",
        "download_pdf": "📄 Download Executive PDF",
        "download_csv": "📊 Download Excel (CSV)",
        "report_error": "⚠️ Install 'reportlab' to generate PDFs."
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
st.set_page_config(page_title="Nexus Data Studios", layout="wide", page_icon="🔷")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    
    .main-header {
        background: linear-gradient(90deg, #0F172A 0%, #334155 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; 
        font-size: 3.5rem; 
        padding-bottom: 0px;
        line-height: 1.1;
    }
    .sub-header {
        color: #64748B;
        font-size: 1rem; 
        font-weight: 400;
        margin-top: 5px;
        margin-bottom: 25px;
        letter-spacing: 1px;
    }
    
    .metric-card {
        background-color: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 24px; text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {font-size: 2rem; font-weight: 700; color: #0F172A;}
    .metric-label {font-size: 0.875rem; color: #64748B; font-weight: 500;}
    .stButton > button {border-radius: 8px; font-weight: 600;}
    [data-testid="stSidebar"] {background-color: #F8FAFC; border-right: 1px solid #E2E8F0;}
    
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# =======================================================
# 3. GESTIÓN DE IDIOMA Y LOGIN 🔐
# =======================================================
if "language" not in st.session_state: st.session_state.language = "ES"
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""

if "presupuestos" not in st.session_state: 
    st.session_state.presupuestos = {cat: 0.0 for cat in CATEGORIAS["ES"]} 

if 'ticket_bytes' not in st.session_state:
    st.session_state.ticket_bytes = None

def login():
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        lang_sel = st.selectbox("Language / Idioma", ["Español", "English"], key="lang_login")
        st.session_state.language = "ES" if lang_sel == "Español" else "EN"
        t = TEXTOS[st.session_state.language]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #0F172A; font-size: 2.5rem;'>SmartReceipt Enterprise</h1>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; color: #64748B;'>by 🔷 Nexus Data Studios</h5>", unsafe_allow_html=True)
        
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
# 5. GENERADOR DE REPORTES PDF 📄
# =======================================================
def generar_reporte_pdf(df_datos, usuario):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # 1. ENCABEZADO
    title_style = styles['Title']
    title_style.textColor = colors.HexColor('#0F172A')
    elements.append(Paragraph("Nexus Data Studios - Informe de Gastos", title_style))
    elements.append(Spacer(1, 12))
    
    normal_style = styles['Normal']
    elements.append(Paragraph(f"<b>Usuario:</b> {usuario}", normal_style))
    elements.append(Paragraph(f"<b>Fecha de Emisión:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
    elements.append(Paragraph(f"<b>Total del Periodo:</b> ${df_datos['Monto'].sum():,.2f}", normal_style))
    elements.append(Spacer(1, 20))
    
    # 2. RESUMEN POR CATEGORÍA (TABLA DE CONTADOR)
    elements.append(Paragraph("<b>Resumen Fiscal por Categoría</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    resumen = df_datos.groupby('Categoría')['Monto'].sum().reset_index()
    data_resumen = [['Categoría', 'Monto Total']] + resumen.values.tolist()
    
    t_resumen = Table(data_resumen, colWidths=[300, 150])
    t_resumen.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 1, colors.white)
    ]))
    elements.append(t_resumen)
    elements.append(Spacer(1, 20))

    # 3. DETALLE DE TRANSACCIONES
    elements.append(Paragraph("<b>Detalle de Transacciones</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    # Seleccionar solo columnas relevantes
    df_detalle = df_datos[['Fecha', 'Comercio', 'Categoría', 'Monto']].copy()
    data_detalle = [['Fecha', 'Comercio', 'Categoría', 'Monto']] + df_detalle.values.tolist()
    
    t_detalle = Table(data_detalle, colWidths=[80, 150, 150, 80])
    t_detalle.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#64748B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    elements.append(t_detalle)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# =======================================================
# 6. FUNCIONES CORE
# =======================================================
def procesar_imagen_opencv(imagen_pil):
    try:
        img_np = np.array(imagen_pil)
        if img_np.shape[-1] == 4: img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else: img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return Image.fromarray(sharpened)
    except:
        return imagen_pil 

def limpiar_json(texto_sucio):
    inicio = texto_sucio.find('{')
    fin = texto_sucio.rfind('}')
    if inicio != -1 and fin != -1: return texto_sucio[inicio:fin+1]
    else: return texto_sucio

def analizar_ticket(imagen_pil, idioma_actual):
    try:
        mods = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        modelo = next((m for m in mods if 'flash' in m and '1.5' in m), mods[0] if mods else "gemini-1.5-flash")
    except: modelo = "gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(modelo)
        cats_str = ", ".join(CATS_ACTUALES)
        instrucciones_idioma = "Output values in ENGLISH." if idioma_actual == "EN" else "Valores de texto en ESPAÑOL."

        prompt = f"""
        Actúa como experto OCR. {instrucciones_idioma}
        LISTA DE CATEGORÍAS: [{cats_str}]
        
        INSTRUCCIONES:
        1. Extrae: Comercio, Fecha (DD/MM/AAAA), Total, Hora.
        2. DIRECCIÓN: Intenta extraer la dirección si es legible.
        3. LAT/LON: Déjalos en 0.0, ya no son necesarios visualizar.
        
        JSON OBLIGATORIO: 
        {{"comercio": "Nombre", "total": 0.00, "fecha": "DD/MM/AAAA", "hora": "HH:MM", "ubicacion": "Dirección", "latitud": 0.0, "longitud": 0.0, "categoria": "Texto", "detalles": "Texto"}}
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
        if val is None or val == "" or str(val).strip().lower() == "null": return 0.0
        return float(val)
    except: return 0.0

# =======================================================
# 7. DASHBOARD & UI
# =======================================================
df_local = pd.DataFrame(st.session_state['gastos'])
df_filtrado = pd.DataFrame()

if not df_local.empty:
    for c in ['lat','lon','Monto']:
        if c in df_local.columns: df_local[c] = pd.to_numeric(df_local[c], errors='coerce').fillna(0.0)
    df_local['Fecha_dt'] = pd.to_datetime(df_local['Fecha'], dayfirst=True, errors='coerce')
    df_local['Mes_Año'] = df_local['Fecha_dt'].dt.strftime('%Y-%m')

# --- SIDEBAR (CONFIG, FILTROS Y REPORTES) ---
with st.sidebar:
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 15px; border: 1px solid #e2e8f0; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h3 style="margin:0; color: #0F172A;">👤 {st.session_state.username}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    lang_side = st.selectbox("🌐 Language", ["Español", "English"], index=0 if st.session_state.language=="ES" else 1)
    if (lang_side == "Español" and st.session_state.language != "ES") or (lang_side == "English" and st.session_state.language != "EN"):
        st.session_state.language = "ES" if lang_side == "Español" else "EN"
        st.rerun()

    st.markdown("---")
    with st.expander(T['budget_set']):
        presupuesto_total_calc = 0.0
        for cat in CATS_ACTUALES:
            val_prev = st.session_state.presupuestos.get(cat, 0.0)
            nuevo_val = st.number_input(f"{cat}", min_value=0.0, value=val_prev, step=100.0)
            st.session_state.presupuestos[cat] = nuevo_val
            presupuesto_total_calc += nuevo_val
    st.session_state.budget = presupuesto_total_calc

    st.markdown(f"### {T['filters']}")
    if not df_local.empty:
        opts_mes = sorted([x for x in df_local['Mes_Año'].unique() if str(x) != 'nan'], reverse=True)
        sel_mes = st.multiselect(T['period'], opts_mes)
        opts_cat = sorted([str(x) for x in df_local['Categoría'].unique() if x])
        sel_cat = st.multiselect(T['category'], opts_cat)
        
        df_filtrado = df_local.copy()
        if sel_mes: df_filtrado = df_filtrado[df_filtrado['Mes_Año'].isin(sel_mes)]
        if sel_cat: df_filtrado = df_filtrado[df_filtrado['Categoría'].isin(sel_cat)]
    
    # --- NUEVA SECCIÓN DE REPORTES EN SIDEBAR ---
    st.markdown("---")
    st.markdown(f"### {T['report_title']}")
    if not df_filtrado.empty:
        # BOTÓN 1: PDF EJECUTIVO
        if HAS_REPORTLAB:
            pdf_data = generar_reporte_pdf(df_filtrado, st.session_state.username)
            st.download_button(label=T['download_pdf'], data=pdf_data, file_name="Reporte_Nexus.pdf", mime="application/pdf")
        else:
            st.warning(T['report_error'])
        
        # BOTÓN 2: EXCEL / CSV
        csv_data = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(label=T['download_csv'], data=csv_data, file_name="Gastos_Nexus.csv", mime="text/csv")
    else:
        st.caption("Filtra datos para generar reportes.")

    st.markdown("---")
    if st.button(T['logout'], use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# --- BRANDING ---
st.markdown('<h1 class="main-header">SmartReceipt Enterprise</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">by 🔷 <b>Nexus Data Studios</b></div>', unsafe_allow_html=True)

# TABS PRINCIPALES
tab_nuevo, tab_dashboard, tab_chat = st.tabs([T['tab1'], T['tab2'], T['tab3']])

# =======================================================
# TAB 1: DIGITALIZACIÓN
# =======================================================
with tab_nuevo:
    if 'temp_data' not in st.session_state:
        # FASE 1: SUBIR Y ANALIZAR
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### 1. Input Source")
            
            # --- EL CARGADOR QUE GUARDA EN LA BÓVEDA ---
            uploaded_file = st.file_uploader(T['upload_label'], type=["jpg","png","jpeg","webp"], key="final_uploader")
            
            # SI HAY ARCHIVO NUEVO, GUARDAR BYTES INMEDIATAMENTE EN LA BÓVEDA
            if uploaded_file is not None:
                st.session_state.ticket_bytes = uploaded_file.getvalue()
            
            st.markdown("---")
            if st.button(T['manual_btn'], use_container_width=True):
                 st.session_state['temp_data'] = {
                    "comercio": "", "total": 0.0, "fecha": datetime.now().strftime("%d/%m/%Y"),
                    "hora": datetime.now().strftime("%H:%M"), "categoria": CATS_ACTUALES[0],
                    "ubicacion": "", "detalles": "Manual", "latitud": 0.0, "longitud": 0.0
                }
                 st.rerun()

        with col2:
            st.markdown("#### 2. Preview & Action")
            # --- VERIFICAMOS LA BÓVEDA, NO EL CARGADOR ---
            if st.session_state.ticket_bytes is not None:
                try:
                    image_stream = io.BytesIO(st.session_state.ticket_bytes)
                    img_pil = Image.open(image_stream)
                    img_proc = procesar_imagen_opencv(img_pil)
                    st.caption(T['preview_label'])
                    st.image(img_proc, use_container_width=True, output_format="JPEG")
                    
                    if st.button(T['analyze_btn'], type="primary", use_container_width=True):
                        with st.spinner("Nexus AI Processing..."):
                            txt, mod = analizar_ticket(img_proc, st.session_state.language)
                            txt_limpio = limpiar_json(txt)
                            if "Error" in txt: st.error(txt)
                            else:
                                try:
                                    st.session_state['temp_data'] = json.loads(txt_limpio)
                                    st.session_state.ticket_bytes = None 
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error parsing: {e}")
                except Exception as e:
                    st.error(f"Error processing stored image: {e}")
                    st.session_state.ticket_bytes = None
            else:
                st.info(T['upload_label'])

    else:
        # FASE 2: VALIDACIÓN
        st.markdown(f"### {T['validation_title']}")
        data = st.session_state['temp_data']
        
        with st.container(border=True):
            c1,c2 = st.columns(2)
            vc = c1.text_input(T['commerce'], data.get("comercio",""), key="val_com")
            try: val_monto = safe_float(str(data.get("total",0)).replace("$","").replace(",",""))
            except: val_monto = 0.0
            vm = c2.number_input("Total ($)", value=val_monto, key="val_tot")

            c3,c4,c5 = st.columns(3)
            vf = c3.text_input("Date (DD/MM/YYYY)", data.get("fecha",""), key="val_date")
            vh = c4.text_input("Time", data.get("hora", "00:00"), key="val_time")
            
            cat_def = data.get("categoria","Misc")
            idx = 0
            if cat_def in CATS_ACTUALES:
                idx = CATS_ACTUALES.index(cat_def)
            else: idx = len(CATS_ACTUALES) - 1
            
            vcat = c5.selectbox(T['category'], CATS_ACTUALES, index=idx, key="val_cat")
            
            with st.expander("📝 Details", expanded=True):
                vu = st.text_input("Location", data.get("ubicacion",""), key="val_loc")
                vdet = st.text_input("Details", data.get("detalles",""), key="val_det")
                vlat = 0.0 
                vlon = 0.0

            col_btn1, col_btn2 = st.columns([1,1])
            with col_btn1:
                if st.button("❌ Cancelar / Volver", use_container_width=True):
                    del st.session_state['temp_data']
                    st.rerun()
            with col_btn2:
                if st.button(T['save_btn'], type="primary", use_container_width=True):
                    nuevo = {"Usuario": st.session_state.username, "Fecha": vf, "Hora": vh, "Comercio": vc, "Monto": vm, "Ubicación": vu, "lat": vlat, "lon": vlon, "Categoría": vcat, "Detalles": vdet}
                    st.session_state['gastos'].append(nuevo)
                    hoja = get_google_sheet()
                    if hoja:
                        try: hoja.append_row(list(nuevo.values()))
                        except: pass
                    del st.session_state['temp_data']
                    st.rerun()

# =======================================================
# TAB 2: DASHBOARD
# =======================================================
with tab_dashboard:
    if not df_filtrado.empty:
        # MÉTRICAS GENERALES
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">{T["total_label"]}</div><div class="metric-value" style="color:#0F172A">${df_filtrado["Monto"].sum():,.0f}</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">{T["trans_label"]}</div><div class="metric-value" style="color:#3B82F6">{len(df_filtrado)}</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">{T["avg_label"]}</div><div class="metric-value" style="color:#F59E0B">${df_filtrado["Monto"].mean():,.0f}</div></div>', unsafe_allow_html=True)
        with m4:
            top_cat = df_filtrado.groupby('Categoría')['Monto'].sum().idxmax() if not df_filtrado.empty else "-"
            st.markdown(f'<div class="metric-card"><div class="metric-label">{T["max_label"]}</div><div class="metric-value" style="color:#EF4444; font-size:1.5rem">{top_cat}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # HIGHLIGHTS
        st.markdown(f"### {T['highlights_title']}")
        hc1, hc2, hc3 = st.columns(3)
        
        idx_max = df_filtrado['Monto'].idxmax()
        row_max = df_filtrado.loc[idx_max]
        with hc1: st.info(f"{T['highlight_max']}:\n\n${row_max['Monto']:,.2f} - **{row_max['Comercio']}**")
        
        cat_top = df_filtrado.groupby('Categoría')['Monto'].sum().idxmax()
        monto_cat = df_filtrado.groupby('Categoría')['Monto'].sum().max()
        with hc2: st.success(f"{T['highlight_top']}:\n\n**{cat_top}** (${monto_cat:,.2f}).")
        
        mask_serv = df_filtrado['Categoría'].str.contains("Servicios|Utilities|Telefonía|Phone", case=False, na=False)
        gastos_serv = df_filtrado[mask_serv]
        total_serv = gastos_serv['Monto'].sum() if not gastos_serv.empty else 0.0
        with hc3: st.warning(f"{T['highlight_serv']}:\n\nTotal: **${total_serv:,.2f}**")

        st.markdown("---")
        
        # --- GRÁFICO LAYERED ---
        st.markdown(f"##### {T['chart_budget_title']}")
        try:
            gastos_por_cat = df_filtrado.groupby('Categoría')['Monto'].sum().reset_index()
            gastos_por_cat.columns = ['Categoría', 'Gasto Real']
            df_presupuestos = pd.DataFrame(list(st.session_state.presupuestos.items()), columns=['Categoría', 'Presupuesto'])
            df_final = pd.merge(df_presupuestos, gastos_por_cat, on='Categoría', how='left').fillna(0)

            base = alt.Chart(df_final).encode(x=alt.X('Categoría', axis=alt.Axis(labelAngle=-45)))
            bar_presupuesto = base.mark_bar(color='#E2E8F0').encode(y='Presupuesto', tooltip=['Categoría', 'Presupuesto'])
            bar_gasto = base.mark_bar(color='#3B82F6', opacity=0.9).encode(y='Gasto Real', tooltip=['Categoría', 'Gasto Real'])
            
            chart_layer = (bar_presupuesto + bar_gasto).properties(height=350)
            st.altair_chart(chart_layer, use_container_width=True)
            st.caption("ℹ️ Barra Gris = Presupuesto | Barra Azul = Gasto Real")
        except Exception as e:
            st.warning("⚠️ Datos insuficientes para gráfico presupuestal.")

        # Gráficos Secundarios
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

        st.markdown(f"### {T['delete_title']}")
        st.caption(T['delete_caption'])
        opciones_borrar = {f"{i} | {r['Fecha']} - {r['Comercio']} (${r['Monto']})": i for i, r in df_filtrado.iterrows()}
        c_del1, c_del2 = st.columns([3,1])
        with c_del1: 
            sel_del = st.selectbox(T['delete_select'], list(opciones_borrar.keys()), key="sel_delete")
        with c_del2: 
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(T['delete_btn'], type="primary", key="btn_delete"):
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

# =======================================================
# TAB 3: CHAT
# =======================================================
with tab_chat:
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input(T['chat_placeholder'], key="chat_input"):
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role":"user", "content":q})
        if df_filtrado.empty: r = "No data."
        else:
            with st.spinner("AI Thinking..."): 
                r = consultar_chat_financiero(q, df_filtrado, st.session_state.language) 
            with st.chat_message("assistant"): st.markdown(r)
            st.session_state['chat_history'].append({"role":"assistant", "content":r})

# FOOTER
st.markdown("<div style='text-align: center; margin-top: 50px; color: #94a3b8; font-size: 12px;'>SmartReceipt Enterprise by Nexus Data Studios © 2026</div>", unsafe_allow_html=True)