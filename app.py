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
from datetime import datetime, timedelta
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
# 1. CEREBRO BILINGÜE (TEXTOS) - V47 ANALYTICS
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
        "period": "📅 Periodo Base",
        "comp_period": "🆚 Comparar contra (Opcional)",
        "category": "🏷️ Categoría",
        "commerce": "🏪 Comercio",
        "logout": "Cerrar Sesión",
        "tab1": "📸 Digitalizar Ticket",
        "tab2": "📈 Master Analytics",
        "tab3": "💬 AI Assistant",
        "upload_label": "📂 Toca para usar CÁMARA o GALERÍA",
        "manual_btn": "✍️ Captura Manual",
        "manual_info": "¿Gasto sin comprobante?",
        "analyze_btn": "⚡ PROCESAR TICKET",
        "save_btn": "💾 Guardar Transacción",
        "validation_title": "✅ Validar Información Extraída",
        "highlights_title": "💡 Insights Financieros",
        "highlight_max": "💸 Compra más grande",
        "highlight_top": "🛍️ Categoría Top",
        "highlight_serv": "⚡ Servicios Básicos",
        "budget_set": "⚙️ Configurar Presupuestos",   
        "budget_used": "Consumido Global",             
        "total_label": "Gasto Total",
        "trans_label": "Transacciones",
        "avg_label": "Ticket Promedio",
        "max_label": "Mayor Gasto",
        "chart_budget_title": "📊 Control Presupuestal", 
        "delete_title": "🗑️ Gestión de Registros",
        "delete_caption": "Selecciona un registro para eliminarlo permanentemente.",
        "delete_select": "Seleccionar Registro a Eliminar",
        "delete_btn": "Eliminar Registro",
        "delete_success": "Registro eliminado de la Nube",
        "chat_placeholder": "Ej: ¿En qué gasté más este mes?",
        "legal_privacy": "**AVISO DE PRIVACIDAD:** Sus datos son usados para gestión de gastos.",
        "legal_terms": "**TÉRMINOS:** Uso bajo su responsabilidad. IA puede cometer errores.",
        "preview_label": "Vista previa (Lista para IA):",
        "report_title": "📑 Centro de Reportes",
        "download_pdf": "📄 Descargar PDF Ejecutivo",
        "download_csv": "📊 Descargar Excel (CSV)",
        "report_error": "⚠️ Instala 'reportlab' para generar PDFs.",
        "type_label": "Tipo de Movimiento",
        "income": "Ingreso",
        "expense": "Gasto",
        "balance_label": "Flujo Neto (Saldo)",
        "forecast_title": "🔮 Pronóstico de Flujo (AI Forecast)",
        "waterfall_title": "🌊 Cascada de Flujo de Efectivo",
        "chart_pie_title": "🍩 Distribución de Gastos (Top Categorías)",
        "chart_trend_title": "📉 Tendencia de Gasto (Día a Día)",
        "chart_cat_compare": "📊 Comparativa por Categoría",
        "metric_vs": "vs periodo anterior"
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
        "period": "📅 Base Period",
        "comp_period": "🆚 Compare with (Optional)",
        "category": "🏷️ Category",
        "commerce": "🏪 Merchant",
        "logout": "Logout",
        "tab1": "📸 Digitize Receipt",
        "tab2": "📈 Master Analytics",
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
        "chart_budget_title": "📊 Budget Control", 
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
        "report_error": "⚠️ Install 'reportlab' to generate PDFs.",
        "type_label": "Transaction Type",
        "income": "Income",
        "expense": "Expense",
        "balance_label": "Net Cash Flow",
        "forecast_title": "🔮 Cash Flow Forecast (AI)",
        "waterfall_title": "🌊 Cash Flow Waterfall",
        "chart_pie_title": "🍩 Expense Distribution",
        "chart_trend_title": "📉 Spending Trend (Day by Day)",
        "chart_cat_compare": "📊 Category Comparison",
        "metric_vs": "vs previous period"
    }
}

CATEGORIAS = {
    "ES": [
        "Alimentos y Supermercado", "Restaurantes y Bares", "Gasolina y Transporte",
        "Salud y Farmacia", "Hogar y Muebles", "Servicios (Luz/Agua/Internet)", 
        "Telefonía", "Ropa y Calzado", "Electrónica", "Entretenimiento", 
        "Educación", "Mascotas", "Regalos", "Viajes", "Suscripciones",
        "Cuidado Personal", "Deportes", "Oficina", "Mantenimiento Auto", 
        "Impuestos y Predial", "Varios", "Nómina/Salario", "Ventas", "Otros Ingresos"
    ],
    "EN": [
        "Groceries & Supermarket", "Restaurants & Bars", "Gas & Transport",
        "Health & Pharmacy", "Home & Furniture", "Utilities (Water/Electric)", 
        "Phone & Internet", "Clothing", "Electronics", "Entertainment", 
        "Education", "Pets", "Gifts", "Travel", "Subscriptions",
        "Personal Care", "Sports", "Office", "Car Maintenance", 
        "Taxes", "Misc", "Salary", "Sales", "Other Income"
    ]
}

# =======================================================
# 2. CONFIGURACIÓN Y ESTILOS UI 🎨
# =======================================================
st.set_page_config(page_title="Nexus Analytics", layout="wide", page_icon="🔷")

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
        st.markdown("<h1 style='text-align: center; color: #0F172A; font-size: 2.5rem;'>Nexus Analytics</h1>", unsafe_allow_html=True)
        
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
                hoja.insert_row(["Usuario", "Tipo", "Fecha", "Hora", "Comercio", "Monto", "Ubicación", "lat", "lon", "Categoría", "Detalles"], 1)
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
def generar_reporte_pdf(df_datos, usuario, periodo_texto, presupuestos_dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # 1. ENCABEZADO
    title_style = styles['Title']
    title_style.textColor = colors.HexColor('#0F172A')
    elements.append(Paragraph("Nexus Data Studios - Informe Financiero", title_style))
    elements.append(Spacer(1, 12))
    
    normal_style = styles['Normal']
    elements.append(Paragraph(f"<b>Usuario:</b> {usuario}", normal_style))
    elements.append(Paragraph(f"<b>Periodo:</b> {periodo_texto}", normal_style))
    elements.append(Paragraph(f"<b>Emisión:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
    
    # Cálculos
    ingresos = df_datos[df_datos['Tipo'] == 'Ingreso']['Monto'].sum()
    gastos = df_datos[df_datos['Tipo'] == 'Gasto']['Monto'].sum()
    balance = ingresos - gastos
    
    elements.append(Paragraph(f"<b>Flujo Neto (Saldo):</b> ${balance:,.2f}", normal_style))
    elements.append(Spacer(1, 20))
    
    # 2. TABLA BALANCE
    data_resumen = [['Concepto', 'Monto']]
    data_resumen.append(['Total Ingresos', f"${ingresos:,.2f}"])
    data_resumen.append(['Total Gastos', f"-${gastos:,.2f}"])
    data_resumen.append(['SALDO FINAL', f"${balance:,.2f}"])
    
    t_resumen = Table(data_resumen, colWidths=[200, 150])
    t_resumen.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.white),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#CBD5E1')),
    ]))
    elements.append(t_resumen)
    elements.append(Spacer(1, 20))

    # 3. DETALLE
    elements.append(Paragraph("<b>Detalle de Movimientos</b>", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    df_detalle = df_datos[['Fecha', 'Tipo', 'Comercio', 'Categoría', 'Monto']].copy()
    data_detalle = [['Fecha', 'Tipo', 'Comercio', 'Categoría', 'Monto']] + df_detalle.values.tolist()
    
    t_detalle = Table(data_detalle, colWidths=[60, 60, 120, 120, 70])
    t_detalle.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#64748B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
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
        2. TIPO: Asume siempre que es "Gasto" (Expense) a menos que diga explícitamente "Depósito" o "Nómina".
        
        JSON OBLIGATORIO: 
        {{"tipo": "Gasto", "comercio": "Nombre", "total": 0.00, "fecha": "DD/MM/AAAA", "hora": "HH:MM", "ubicacion": "Dirección", "latitud": 0.0, "longitud": 0.0, "categoria": "Texto", "detalles": "Texto"}}
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
        prompt = f"Role: Financial Data Scientist. {lang_prompt}. Data: \n---\n{datos_csv}\n---\nQuery: {pregunta}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Error Chat: {e}"

def safe_float(val):
    try:
        if val is None or val == "" or str(val).strip().lower() == "null": return 0.0
        return float(val)
    except: return 0.0

# =======================================================
# 7. DASHBOARD & UI (V47 MASTER ANALYTICS)
# =======================================================
df_local = pd.DataFrame(st.session_state['gastos'])
df_base = pd.DataFrame()
df_comp = pd.DataFrame() # DataFrame de comparación

if not df_local.empty:
    for c in ['lat','lon','Monto']:
        if c in df_local.columns: df_local[c] = pd.to_numeric(df_local[c], errors='coerce').fillna(0.0)
    df_local['Fecha_dt'] = pd.to_datetime(df_local['Fecha'], dayfirst=True, errors='coerce')
    df_local['Mes_Año'] = df_local['Fecha_dt'].dt.strftime('%Y-%m')
    df_local['Dia'] = df_local['Fecha_dt'].dt.day # Para tendencias
    
    if 'Tipo' not in df_local.columns: df_local['Tipo'] = 'Gasto'

# --- SIDEBAR: FILTROS AVANZADOS V47 ---
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
    st.subheader(T['filters'])
    
    if not df_local.empty:
        # 1. Selector Periodo Base (Principal)
        opts_mes = sorted([x for x in df_local['Mes_Año'].unique() if str(x) != 'nan'], reverse=True)
        mes_base = st.selectbox(T['period'], opts_mes, index=0)
        
        # 2. Selector Periodo Comparación (Opcional)
        opts_comp = ["Ninguno"] + [m for m in opts_mes if m != mes_base]
        mes_comp = st.selectbox(T['comp_period'], opts_comp, index=0)
        
        # Filtrado de DataFrames
        df_base = df_local[df_local['Mes_Año'] == mes_base].copy()
        
        if mes_comp != "Ninguno":
            df_comp = df_local[df_local['Mes_Año'] == mes_comp].copy()
            st.info(f"📅 Comparando: **{mes_base}** vs **{mes_comp}**")
        else:
            st.caption("Selecciona otro mes para ver comparativas.")

    st.markdown("---")
    # Botón de Reporte (Descarga lo que ves en el Dashboard)
    st.markdown(f"### {T['report_title']}")
    if not df_base.empty:
        if HAS_REPORTLAB:
            periodo_str = mes_base if mes_comp == "Ninguno" else f"{mes_base} vs {mes_comp}"
            pdf_data = generar_reporte_pdf(df_base, st.session_state.username, periodo_str, st.session_state.presupuestos)
            st.download_button(label=T['download_pdf'], data=pdf_data, file_name="Reporte_Nexus.pdf", mime="application/pdf")
        csv_data = df_base.to_csv(index=False).encode('utf-8')
        st.download_button(label=T['download_csv'], data=csv_data, file_name="Gastos_Nexus.csv", mime="text/csv")

    st.markdown("---")
    if st.button(T['logout'], use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# --- BRANDING ---
st.markdown(f'<h1 class="main-header">Nexus Analytics <span style="font-size:1.5rem; color:#64748B">V47</span></h1>', unsafe_allow_html=True)

# TABS
tab_nuevo, tab_dashboard, tab_chat = st.tabs([T['tab1'], T['tab2'], T['tab3']])

# =======================================================
# TAB 2: MASTER ANALYTICS (EL CEREBRO VISUAL)
# =======================================================
with tab_dashboard:
    if df_base.empty:
        st.info("No hay datos para el periodo seleccionado.")
    else:
        # --- 1. TARJETAS DE KPIs CON DELTAS ---
        # Calculamos totales Base
        ingreso_base = df_base[df_base['Tipo']=='Ingreso']['Monto'].sum()
        gasto_base = df_base[df_base['Tipo']=='Gasto']['Monto'].sum()
        neto_base = ingreso_base - gasto_base
        
        # Calculamos totales Comparación (si existe)
        delta_ingreso = None
        delta_gasto = None
        delta_neto = None
        
        if not df_comp.empty:
            ingreso_comp = df_comp[df_comp['Tipo']=='Ingreso']['Monto'].sum()
            gasto_comp = df_comp[df_comp['Tipo']=='Gasto']['Monto'].sum()
            neto_comp = ingreso_comp - gasto_comp
            
            # Cálculo de porcentajes
            if ingreso_comp > 0: delta_ingreso = f"{((ingreso_base - ingreso_comp)/ingreso_comp)*100:.1f}%"
            if gasto_comp > 0: delta_gasto = f"{((gasto_base - gasto_comp)/gasto_comp)*100:.1f}%"
            delta_neto = f"${neto_base - neto_comp:,.0f}" # Diferencia monetaria para el neto
        
        # Renderizamos Métricas Nativas de Streamlit (Se ven muy pro)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(label=T['income'], value=f"${ingreso_base:,.0f}", delta=delta_ingreso)
        k2.metric(label=T['total_label'], value=f"${gasto_base:,.0f}", delta=delta_gasto, delta_color="inverse") # Rojo si sube el gasto
        k3.metric(label=T['balance_label'], value=f"${neto_base:,.0f}", delta=delta_neto)
        
        # Burn Rate simple
        dias_trans = df_base['Dia'].max()
        burn = gasto_base / dias_trans if dias_trans > 0 else 0
        k4.metric(label="Burn Rate (Diario)", value=f"${burn:,.0f}", help="Gasto promedio por día")
        
        st.markdown("---")

        # --- 2. GRÁFICO DE DONA (GASTOS) ---
        col_graf1, col_graf2 = st.columns([1, 2])
        
        with col_graf1:
            st.subheader(T['chart_pie_title'])
            # Filtramos solo gastos y agrupamos
            df_pie = df_base[df_base['Tipo']=='Gasto'].groupby('Categoría')['Monto'].sum().reset_index()
            
            # Gráfico de Dona con Altair
            base = alt.Chart(df_pie).encode(theta=alt.Theta("Monto", stack=True))
            pie = base.mark_arc(innerRadius=80, outerRadius=120).encode(
                color=alt.Color("Categoría", scale=alt.Scale(scheme='category20b'), legend=None), # Sin leyenda para limpieza
                order=alt.Order("Monto", sort="descending"),
                tooltip=["Categoría", alt.Tooltip("Monto", format="$,.2f")]
            )
            # Texto en el centro (Total Gastos)
            text = base.mark_text(radius=0).encode(
                text=alt.value(f"${gasto_base/1000:.1f}k"),
                size=alt.value(20),
                color=alt.value("#334155")
            )
            st.altair_chart(pie + text, use_container_width=True)
            # Pequeña leyenda manual abajo para las Top 3
            if not df_pie.empty:
                top3 = df_pie.nlargest(3, 'Monto')
                for i, r in top3.iterrows():
                    st.caption(f"🔹 **{r['Categoría']}**: ${r['Monto']:,.0f}")

        # --- 3. TENDENCIA COMPARATIVA (LÍNEAS) ---
        with col_graf2:
            st.subheader(T['chart_trend_title'])
            
            # Preparamos datos para la línea
            df_line_base = df_base[df_base['Tipo']=='Gasto'].groupby('Dia')['Monto'].sum().reset_index()
            df_line_base['Periodo'] = mes_base
            
            datos_trend = df_line_base
            
            if not df_comp.empty:
                df_line_comp = df_comp[df_comp['Tipo']=='Gasto'].groupby('Dia')['Monto'].sum().reset_index()
                df_line_comp['Periodo'] = mes_comp
                datos_trend = pd.concat([df_line_base, df_line_comp])
            
            # Gráfico de Líneas Multiserie
            chart_trend = alt.Chart(datos_trend).mark_line(point=True, interpolate='monotone').encode(
                x=alt.X('Dia', title='Día del Mes'),
                y=alt.Y('Monto', title='Gasto Diario ($)'),
                color=alt.Color('Periodo', legend=alt.Legend(title="Mes", orient="top")),
                tooltip=['Periodo', 'Dia', alt.Tooltip('Monto', format="$,.2f")]
            ).properties(height=350)
            
            st.altair_chart(chart_trend, use_container_width=True)

        st.markdown("---")

        # --- 4. ANÁLISIS POR CATEGORÍA (COMPARATIVO) ---
        if not df_comp.empty:
            st.subheader(T['chart_cat_compare'])
            # Preparamos datos conjuntos
            cat_base = df_base[df_base['Tipo']=='Gasto'].groupby('Categoría')['Monto'].sum().reset_index()
            cat_base['Periodo'] = mes_base
            
            cat_comp = df_comp[df_comp['Tipo']=='Gasto'].groupby('Categoría')['Monto'].sum().reset_index()
            cat_comp['Periodo'] = mes_comp
            
            cat_total = pd.concat([cat_base, cat_comp])
            
            chart_bar = alt.Chart(cat_total).mark_bar().encode(
                x=alt.X('Categoría', axis=alt.Axis(labelAngle=-45)),
                y='Monto',
                color='Periodo',
                xOffset='Periodo', # Barras agrupadas lado a lado
                tooltip=['Categoría', 'Periodo', alt.Tooltip('Monto', format="$,.2f")]
            ).properties(height=400)
            st.altair_chart(chart_bar, use_container_width=True)
        else:
             # Si no hay comparación, mostramos el Waterfall original
             st.subheader(T['waterfall_title'])
             try:
                data_waterfall = [{"Concepto": "Ingresos", "Monto": ingreso_base, "Color": "Ingreso"}]
                gastos_cat = df_base[df_base['Tipo'] == 'Gasto'].groupby('Categoría')['Monto'].sum().reset_index()
                for i, row in gastos_cat.iterrows():
                    data_waterfall.append({"Concepto": row['Categoría'], "Monto": -row['Monto'], "Color": "Gasto"})
                data_waterfall.append({"Concepto": "Saldo Final", "Monto": neto_base, "Color": "Total"})
                df_wf = pd.DataFrame(data_waterfall)
                wf_chart = alt.Chart(df_wf).mark_bar().encode(
                    x=alt.X('Concepto', sort=None), y='Monto',
                    color=alt.Color('Color', scale={'domain': ['Ingreso', 'Gasto', 'Total'], 'range': ['#10B981', '#EF4444', '#3B82F6']}),
                    tooltip=['Concepto', 'Monto']
                ).properties(height=400)
                st.altair_chart(wf_chart, use_container_width=True)
             except: pass

# =======================================================
# TAB 1: DIGITALIZACIÓN (CON SOPORTE DE TIPO)
# =======================================================
with tab_nuevo:
    if 'temp_data' not in st.session_state:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("#### 1. Input Source")
            uploaded_file = st.file_uploader(T['upload_label'], type=["jpg","png","jpeg","webp"], key="final_uploader")
            if uploaded_file is not None:
                st.session_state.ticket_bytes = uploaded_file.getvalue()
            
            st.markdown("---")
            if st.button(T['manual_btn'], use_container_width=True):
                 st.session_state['temp_data'] = {
                    "tipo": "Gasto", # Default
                    "comercio": "", "total": 0.0, "fecha": datetime.now().strftime("%d/%m/%Y"),
                    "hora": datetime.now().strftime("%H:%M"), "categoria": CATS_ACTUALES[0],
                    "ubicacion": "", "detalles": "Manual", "latitud": 0.0, "longitud": 0.0
                }
                 st.rerun()

        with col2:
            st.markdown("#### 2. Preview & Action")
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
        st.markdown(f"### {T['validation_title']}")
        data = st.session_state['temp_data']
        
        with st.container(border=True):
            # V46: Selector de Tipo
            col_t1, col_t2 = st.columns([1,3])
            tipo_idx = 0 if data.get("tipo","Gasto") == "Gasto" else 1
            vt = col_t1.selectbox(T['type_label'], ["Gasto", "Ingreso"], index=tipo_idx, key="val_tipo")
            
            vc = col_t2.text_input(T['commerce'], data.get("comercio",""), key="val_com")

            c1,c2,c3 = st.columns(3)
            try: val_monto = safe_float(str(data.get("total",0)).replace("$","").replace(",",""))
            except: val_monto = 0.0
            vm = c1.number_input("Total ($)", value=val_monto, key="val_tot")
            vf = c2.text_input("Date (DD/MM/YYYY)", data.get("fecha",""), key="val_date")
            vh = c3.text_input("Time", data.get("hora", "00:00"), key="val_time")
            
            cat_def = data.get("categoria","Misc")
            idx = 0
            if cat_def in CATS_ACTUALES: idx = CATS_ACTUALES.index(cat_def)
            else: idx = len(CATS_ACTUALES) - 1
            vcat = st.selectbox(T['category'], CATS_ACTUALES, index=idx, key="val_cat")
            
            with st.expander("📝 Details", expanded=True):
                vu = st.text_input("Location", data.get("ubicacion",""), key="val_loc")
                vdet = st.text_input("Details", data.get("detalles",""), key="val_det")

            col_btn1, col_btn2 = st.columns([1,1])
            with col_btn1:
                if st.button("❌ Cancelar", use_container_width=True):
                    del st.session_state['temp_data']
                    st.rerun()
            with col_btn2:
                if st.button(T['save_btn'], type="primary", use_container_width=True):
                    # V46: Estructura actualizada con Tipo en indice 1
                    nuevo = {"Usuario": st.session_state.username, "Tipo": vt, "Fecha": vf, "Hora": vh, "Comercio": vc, "Monto": vm, "Ubicación": vu, "lat": 0.0, "lon": 0.0, "Categoría": vcat, "Detalles": vdet}
                    st.session_state['gastos'].append(nuevo)
                    hoja = get_google_sheet()
                    if hoja:
                        try: hoja.append_row(list(nuevo.values()))
                        except: pass
                    del st.session_state['temp_data']
                    st.rerun()

# =======================================================
# TAB 3: CHAT
# =======================================================
with tab_chat:
    for m in st.session_state['chat_history']:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q := st.chat_input(T['chat_placeholder'], key="chat_input"):
        with st.chat_message("user"): st.markdown(q)
        st.session_state['chat_history'].append({"role":"user", "content":q})
        if df_base.empty: r = "No data."
        else:
             with st.spinner("AI Thinking..."):
                 r = consultar_chat_financiero(q, df_base, st.session_state.language)
             with st.chat_message("assistant"): st.markdown(r)
             st.session_state['chat_history'].append({"role":"assistant", "content":r})

# FOOTER
st.markdown("<div style='text-align: center; margin-top: 50px; color: #94a3b8; font-size: 12px;'>SmartReceipt Enterprise by Nexus Data Studios © 2026</div>", unsafe_allow_html=True)