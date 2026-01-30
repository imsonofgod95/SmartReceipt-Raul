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

# =======================================================
# 1. CONFIGURACIÓN SEGURA (CLOUD READY) ☁️
# =======================================================
st.set_page_config(page_title="SmartReceipt Cloud", layout="wide", page_icon="☁️")

# Intentamos leer la API KEY desde los "Secretos" de Streamlit
# Esto evita que te roben la llave cuando subas el código a GitHub.
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    else:
        st.error("⚠️ Falta la API Key. Por favor configúrala en los 'Secrets' de Streamlit Cloud.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error de configuración: {e}. Si estás en local, crea un archivo .streamlit/secrets.toml")
    st.stop()

ARCHIVO_DB = "historial_gastos.csv"

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

# --- CARGA INICIAL ROBUSTA ---
if 'gastos' not in st.session_state:
    if os.path.exists(ARCHIVO_DB):
        try:
            df_disco = pd.read_csv(ARCHIVO_DB)
            # Migración y limpieza
            if 'lat' not in df_disco.columns: df_disco['lat'] = 0.0 
            if 'lon' not in df_disco.columns: df_disco['lon'] = 0.0
            df_disco['lat'] = pd.to_numeric(df_disco['lat'], errors='coerce').fillna(0.0)
            df_disco['lon'] = pd.to_numeric(df_disco['lon'], errors='coerce').fillna(0.0)
            
            df_disco.to_csv(ARCHIVO_DB, index=False)
            st.session_state['gastos'] = df_disco.to_dict('records')
        except:
            st.session_state['gastos'] = []
    else:
        st.session_state['gastos'] = []

# Inicializar historial del Chatbot
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
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 11)
    clean = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    return Image.fromarray(clean)

# =======================================================
# 3. CONEXIÓN IA (VISIÓN + CHAT)
# =======================================================
def obtener_modelo_valido():
    try:
        lista = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in lista:
            if 'flash' in m and '1.5' in m: return m
        for m in lista:
            if 'pro' in m and '1.5' in m: return m
        return lista[0] if lista else None
    except: return None

def analizar_ticket(imagen_pil):
    nombre = obtener_modelo_valido()
    if not nombre: return "Error: Sin modelos", ""
    try:
        model = genai.GenerativeModel(nombre)
        cats_str = ", ".join(LISTA_CATEGORIAS)
        prompt = f"""
        Analiza imagen. Extrae JSON EXCLUSIVO. NO incluyas formato markdown.
        INSTRUCCIONES UBICACIÓN: Ignora dirección fiscal. Busca SUCURSAL física y estima GPS.
        INSTRUCCIONES CATEGORÍA: Clasifica en una de: [{cats_str}]
        
        JSON:
        {{
            "comercio": "Nombre", "total": 0.00, "fecha": "DD/MM/AAAA",
            "ubicacion": "Sucursal", "latitud": 19.0000, "longitud": -99.0000,
            "categoria": "Texto", "detalles": "Texto"
        }}
        """
        response = model.generate_content([prompt, imagen_pil])
        return response.text, nombre
    except Exception as e: return f"Error: {e}", nombre

def consultar_chat_financiero(pregunta, datos_df):
    """
    Toma todo el historial de gastos y se lo pasa a la IA para que responda preguntas.
    """
    nombre = obtener_modelo_valido()
    try:
        model = genai.GenerativeModel(nombre)
        
        # Convertimos el DataFrame a texto CSV para que la IA lo lea
        datos_csv = datos_df.to_csv(index=False)
        
        prompt_sistema = f"""
        Eres un Asistente Financiero Personal experto.
        
        A continuación tienes el historial de gastos del usuario en formato CSV:
        ---
        {datos_csv}
        ---
        
        INSTRUCCIONES:
        1. Responde a la pregunta del usuario basándote ÚNICAMENTE en estos datos.
        2. Si te piden sumas, promedios o máximos, calcúlalos con precisión.
        3. Sé amable, breve y directo.
        4. Si notas gastos excesivos en "Restaurantes" o "Varios", da un pequeño consejo de ahorro amigable.
        
        PREGUNTA DEL USUARIO: {pregunta}
        """
        
        response = model.generate_content(prompt_sistema)
        return response.text
    except Exception as e:
        return f"Lo siento, tuve un error analizando los datos: {e}"

# =======================================================
# 4. DASHBOARD CON FILTROS
# =======================================================
st.sidebar.header("🔍 Filtros")
df_filtrado = pd.DataFrame()

if st.session_state['gastos']:
    df = pd.DataFrame(st.session_state['gastos'])
    
    if 'Categoría' in df.columns:
        cats = sorted(df['Categoría'].astype(str).unique().tolist())
        sel_cat = st.sidebar.multiselect("Categoría", cats, default=cats)
    else: sel_cat = []
        
    if 'Comercio' in df.columns:
        coms = sorted(df['Comercio'].astype(str).unique().tolist())
        sel_com = st.sidebar.multiselect("Comercio", coms, default=coms)
    else: sel_com = []
    
    if not df.empty:
        mask_cat = df['Categoría'].isin(sel_cat) if 'Categoría' in df.columns else True
        mask_com = df['Comercio'].isin(sel_com) if 'Comercio' in df.columns else True
        df_filtrado = df[mask_cat & mask_com]
else:
    st.sidebar.info("Sube tickets para habilitar filtros.")

# =======================================================
# 5. INTERFAZ PRINCIPAL
# =======================================================
st.title("🧠 SmartReceipt: Cloud Edition")
st.markdown("---")

# AHORA SON 3 PESTAÑAS
tab_nuevo, tab_dashboard, tab_chat = st.tabs(["📸 Nuevo Ticket", "📈 Dashboard", "💬 Chat IA"])

# --- PESTAÑA 1: CARGA ---
with tab_nuevo:
    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        archivo = st.file_uploader("Sube ticket", type=["jpg", "png", "jpeg"])
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, caption="Procesada", use_container_width=True)
            if st.button("🧠 Escanear", type="primary"):
                with st.spinner("Analizando..."):
                    txt, mod = analizar_ticket(img_proc)
                    
                    # AJUSTE ROBUSTO PARA ERROR DE LECTURA
                    try:
                        # Extraer solo el JSON usando Regex por si la IA agrega texto extra
                        match = re.search(r'\{.*\}', txt, re.DOTALL)
                        if match:
                            clean_json = match.group()
                            st.session_state['temp_data'] = json.loads(clean_json)
                            st.toast("Leído", icon="📍")
                        else:
                            st.error("No se detectó un formato JSON válido.")
                    except: 
                        st.error("Error al decodificar la respuesta de la IA.")

    with col_der:
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.form("form_save"):
                st.subheader("Validar")
                c1, c2 = st.columns(2)
                vc = c1.text_input("Comercio", data.get("comercio",""))
                
                # Limpieza de monto previa al número
                monto_raw = str(data.get("total",0)).replace("$","").replace(",","")
                try: vm_f = float(monto_raw)
                except: vm_f = 0.0
                
                vm = c2.number_input("Total", value=vm_f)
                
                c3, c4 = st.columns(2)
                vf = c3.text_input("Fecha", data.get("fecha","Hoy"))
                
                cat_in = data.get("categoria", "Varios")
                idx = LISTA_CATEGORIAS.index(cat_in) if cat_in in LISTA_CATEGORIAS else 19
                vcat = c4.selectbox("Categoría", LISTA_CATEGORIAS, index=idx)
                
                c5, c6, c7 = st.columns([2, 1, 1])
                vu = c5.text_input("Sucursal", data.get("ubicacion",""))
                vlat = c6.number_input("Lat", value=float(data.get("latitud", 19.4326)), format="%.4f")
                vlon = c7.number_input("Lon", value=float(data.get("longitud", -99.1332)), format="%.4f")
                vdet = st.text_input("Detalles", data.get("detalles",""))
                
                if st.form_submit_button("💾 Guardar"):
                    st.session_state['gastos'].append({
                        "Fecha": vf, "Comercio": vc, "Monto": vm, 
                        "Ubicación": vu, "lat": vlat, "lon": vlon,
                        "Categoría": vcat, "Detalles": vdet
                    })
                    pd.DataFrame(st.session_state['gastos']).to_csv(ARCHIVO_DB, index=False)
                    st.success("Guardado")
                    del st.session_state['temp_data']
                    st.rerun()

# --- PESTAÑA 2: DASHBOARD ---
with tab_dashboard:
    if not df_filtrado.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("💰 Total", f"${df_filtrado['Monto'].sum():,.2f}")
        k2.metric("📊 Promedio", f"${df_filtrado['Monto'].mean():,.2f}")
        k3.metric("🧾 Tickets", len(df_filtrado))
        st.divider()
        
        # Mapa
        if 'lat' in df_filtrado.columns:
            map_data = df_filtrado.copy()
            map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce').fillna(0)
            map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce').fillna(0)
            map_data = map_data[(map_data['lat']!=0) & (map_data['lon']!=0)]
            
            if not map_data.empty:
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=10, pitch=40),
                    layers=[pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_color='[255, 75, 75, 200]', get_radius=400, pickable=True)],
                    tooltip={"html": "<b>{Comercio}</b><br/>${Monto}"}
                ))
            else: st.info("Sin datos GPS válidos.")
        
        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.altair_chart(alt.Chart(df_filtrado).mark_arc(innerRadius=60).encode(theta='Monto', color='Categoría', tooltip=['Categoría', 'Monto']), use_container_width=True)
        with g2:
            # Fix Fechas
            df_chart = df_filtrado.copy()
            df_chart['Fecha_dt'] = pd.to_datetime(df_chart['Fecha'], dayfirst=True, errors='coerce')
            df_chart = df_chart.dropna(subset=['Fecha_dt']).sort_values('Fecha_dt')
            if not df_chart.empty:
                st.altair_chart(alt.Chart(df_chart).mark_line(point=True).encode(x=alt.X('Fecha_dt', title='Fecha', axis=alt.Axis(format='%d/%m')), y='Monto', tooltip=['Fecha', 'Monto']), use_container_width=True)
            else: st.caption("Fechas no procesables.")
            
        with st.expander("Ver Datos"):
            st.dataframe(df_filtrado, use_container_width=True)
            if st.button("🗑️ Borrar DB"):
                if os.path.exists(ARCHIVO_DB): os.remove(ARCHIVO_DB)
                st.session_state['gastos'] = []
                st.rerun()

# --- PESTAÑA 3: CHAT IA (NUEVO) ---
with tab_chat:
    st.header("💬 Asistente Financiero")
    st.caption("Pregunta sobre tus gastos. Ej: '¿Cuánto gasté en Gasolina este mes?'")

    # Contenedor del chat
    for mensaje in st.session_state['chat_history']:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    # Input del usuario
    prompt_usuario = st.chat_input("Escribe tu pregunta aquí...")
    
    if prompt_usuario:
        # 1. Mostrar mensaje usuario
        with st.chat_message("user"):
            st.markdown(prompt_usuario)
        st.session_state['chat_history'].append({"role": "user", "content": prompt_usuario})
        
        # 2. Verificar si hay datos
        if not st.session_state['gastos']:
            respuesta = "Aún no tienes tickets registrados. Sube algunos primero."
        else:
            # 3. Procesar con IA
            with st.spinner("Analizando tus finanzas..."):
                df_chat = pd.DataFrame(st.session_state['gastos'])
                respuesta = consultar_chat_financiero(prompt_usuario, df_chat)
        
        # 4. Mostrar respuesta IA
        with st.chat_message("assistant"):
            st.markdown(respuesta)
        st.session_state['chat_history'].append({"role": "assistant", "content": respuesta})