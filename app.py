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
# 2. MÓDULO DE VISIÓN (CORREGIDO - MENOS AGRESIVO)
# =======================================================
def procesar_imagen_opencv(imagen_pil):
    """
    Simplemente mejora el contraste y escala de grises, sin binarizar agresivamente.
    Esto ayuda a Gemini a ver mejor los detalles del texto.
    """
    img_np = np.array(imagen_pil)
    
    # Convertir a BGR si es necesario
    if img_np.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 1. Escala de grises
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Aumentar un poco el contraste (CLAHE) - opcional pero ayuda
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return Image.fromarray(enhanced)

# =======================================================
# 3. CONEXIÓN IA (CORRECCIÓN DINÁMICA DE MODELO Y CUOTA)
# =======================================================
def obtener_modelo_valido():
    """
    Busca dinámicamente el modelo Flash disponible para evitar errores 404.
    Prioriza versiones estables de 1.5 para mantener la cuota gratuita de 1500 RPM.
    """
    try:
        modelos_disp = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Intentamos encontrar gemini-1.5-flash específicamente
        for m in modelos_disp:
            if "gemini-1.5-flash" in m:
                return m
        # Si no existe, devolvemos el primero que soporte contenido
        return modelos_disp[0]
    except:
        # Fallback por si list_models falla
        return "gemini-1.5-flash"

def analizar_ticket(imagen_pil):
    nombre = obtener_modelo_valido()
    try:
        model = genai.GenerativeModel(nombre)
        cats_str = ", ".join(LISTA_CATEGORIAS)
        prompt = f"""
        Analiza esta imagen de un ticket de compra. Extrae la información en formato JSON EXCLUSIVAMENTE.
        
        INSTRUCCIONES UBICACIÓN: Ignora la dirección fiscal corporativa. Busca la dirección de la SUCURSAL física y estima sus coordenadas GPS (lat/lon).
        INSTRUCCIONES CATEGORÍA: Clasifica el gasto en una de estas categorías: [{cats_str}]
        
        Formato de respuesta JSON requerido:
        {{
            "comercio": "Nombre del Comercio", 
            "total": 0.00, 
            "fecha": "DD/MM/AAAA",
            "ubicacion": "Nombre Sucursal o Calle", 
            "latitud": 19.0000, 
            "longitud": -99.0000,
            "categoria": "Categoría elegida", 
            "detalles": "Breve descripción de los items principales"
        }}
        """
        response = model.generate_content([prompt, imagen_pil])
        return response.text, nombre
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return "ERROR_CUOTA: Has excedido el límite por minuto. Espera 60 segundos.", nombre
        if "404" in error_msg:
            return "ERROR_MODELO: No se encontró el modelo en tu región/cuenta.", nombre
        return f"Error API: {e}", nombre

def consultar_chat_financiero(pregunta, datos_df):
    nombre = obtener_modelo_valido()
    try:
        model = genai.GenerativeModel(nombre)
        datos_csv = datos_df.to_csv(index=False)
        
        prompt_sistema = f"""
        Eres un Asistente Financiero Personal experto.
        
        Datos del usuario (CSV):
        ---
        {datos_csv}
        ---
        
        INSTRUCCIONES:
        1. Responde a la pregunta del usuario basándote ÚNICAMENTE en estos datos.
        2. Si piden cálculos, hazlos con precisión.
        3. Sé amable y conciso.
        
        PREGUNTA: {pregunta}
        """
        response = model.generate_content(prompt_sistema)
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⚠️ El asistente está saturado. Por favor espera un momento."
        return f"Error en chat: {e}"

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

tab_nuevo, tab_dashboard, tab_chat = st.tabs(["📸 Nuevo Ticket", "📈 Dashboard", "💬 Chat IA"])

# --- PESTAÑA 1: CARGA ---
with tab_nuevo:
    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        archivo = st.file_uploader("Sube ticket", type=["jpg", "png", "jpeg"])
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, caption="Imagen para análisis", use_container_width=True)
            
            if st.button("🧠 Escanear con IA", type="primary"):
                with st.spinner("Analizando ticket..."):
                    txt, mod = analizar_ticket(img_proc)
                    
                    if "ERROR" in txt:
                        st.error(txt)
                    else:
                        clean = txt.replace("```json", "").replace("```", "").strip()
                        if "{" in clean: 
                            clean = clean[clean.find("{"):clean.rfind("}")+1]
                        
                        try:
                            st.session_state['temp_data'] = json.loads(clean)
                            st.toast("Ticket leído correctamente", icon="✅")
                        except:
                            st.error("Error interpretando la respuesta de la IA.")
                            with st.expander("Ver respuesta cruda"):
                                st.text(txt)

    with col_der:
        if 'temp_data' in st.session_state:
            data = st.session_state['temp_data']
            with st.form("form_save"):
                st.subheader("Verificar Datos")
                
                c1, c2 = st.columns(2)
                vc = c1.text_input("Comercio", data.get("comercio",""))
                
                val_total = data.get("total", 0.0)
                if isinstance(val_total, str):
                    val_total = float(val_total.replace("$","").replace(",",""))
                vm = c2.number_input("Total", value=float(val_total))
                
                c3, c4 = st.columns(2)
                vf = c3.text_input("Fecha", data.get("fecha",""))
                cat_in = data.get("categoria", "Varios")
                idx = LISTA_CATEGORIAS.index(cat_in) if cat_in in LISTA_CATEGORIAS else 19
                vcat = c4.selectbox("Categoría", LISTA_CATEGORIAS, index=idx)
                
                c5, c6, c7 = st.columns([2, 1, 1])
                vu = c5.text_input("Sucursal", data.get("ubicacion",""))
                vlat = c6.number_input("Lat", value=float(data.get("latitud", 0.0)), format="%.4f")
                vlon = c7.number_input("Lon", value=float(data.get("longitud", 0.0)), format="%.4f")
                
                vdet = st.text_input("Detalles", data.get("detalles",""))
                
                if st.form_submit_button("💾 Guardar en Historial"):
                    nuevo_gasto = {
                        "Fecha": vf, "Comercio": vc, "Monto": vm, 
                        "Ubicación": vu, "lat": vlat, "lon": vlon,
                        "Categoría": vcat, "Detalles": vdet
                    }
                    st.session_state['gastos'].append(nuevo_gasto)
                    pd.DataFrame(st.session_state['gastos']).to_csv(ARCHIVO_DB, index=False)
                    st.success("¡Guardado exitosamente!")
                    del st.session_state['temp_data']
                    st.rerun()

# --- PESTAÑA 2: DASHBOARD ---
with tab_dashboard:
    if not df_filtrado.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("💰 Gasto Total", f"${df_filtrado['Monto'].sum():,.2f}")
        k2.metric("📊 Ticket Promedio", f"${df_filtrado['Monto'].mean():,.2f}")
        k3.metric("🧾 Cantidad Tickets", len(df_filtrado))
        st.divider()
        
        if 'lat' in df_filtrado.columns:
            map_data = df_filtrado.copy()
            map_data['lat'] = pd.to_numeric(map_data['lat'], errors='coerce').fillna(0)
            map_data['lon'] = pd.to_numeric(map_data['lon'], errors='coerce').fillna(0)
            map_data = map_data[(map_data['lat']!=0) & (map_data['lon']!=0)]
            
            if not map_data.empty:
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=map_data['lat'].mean(), longitude=map_data['lon'].mean(), zoom=11, pitch=40
                    ),
                    layers=[pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', 
                                      get_color='[255, 75, 75, 200]', get_radius=200, pickable=True)],
                    tooltip={"html": "<b>{Comercio}</b><br/>${Monto}"}
                ))
        
        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Por Categoría")
            st.altair_chart(alt.Chart(df_filtrado).mark_arc(innerRadius=50).encode(
                theta='Monto', color='Categoría', tooltip=['Categoría', 'Monto']), use_container_width=True)
        with g2:
            st.subheader("Evolución Temporal")
            df_chart = df_filtrado.copy()
            df_chart['Fecha_dt'] = pd.to_datetime(df_chart['Fecha'], dayfirst=True, errors='coerce')
            df_chart = df_chart.dropna(subset=['Fecha_dt']).sort_values('Fecha_dt')
            if not df_chart.empty:
                st.altair_chart(alt.Chart(df_chart).mark_line(point=True).encode(
                    x=alt.X('Fecha_dt', title='Fecha'), y='Monto', tooltip=['Fecha', 'Monto', 'Comercio']), use_container_width=True)
            
        with st.expander("📂 Ver Base de Datos Completa"):
            st.dataframe(df_filtrado, use_container_width=True)
            if st.button("🗑️ Borrar Historial"):
                if os.path.exists(ARCHIVO_DB): os.remove(ARCHIVO_DB)
                st.session_state['gastos'] = []
                st.rerun()
    else:
        st.info("No hay gastos registrados.")

# --- PESTAÑA 3: CHAT IA ---
with tab_chat:
    st.header("💬 Asistente Financiero")
    
    # Mostrar el historial guardado
    for mensaje in st.session_state['chat_history']:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    prompt_usuario = st.chat_input("Escribe tu pregunta aquí...")
    
    if prompt_usuario:
        # 1. Mostrar mensaje del usuario inmediatamente
        with st.chat_message("user"):
            st.markdown(prompt_usuario)
        
        # 2. Guardar en el historial
        st.session_state['chat_history'].append({"role": "user", "content": prompt_usuario})
        
        if not st.session_state['gastos']:
            respuesta = "Aún no tienes tickets registrados en la base de datos."
        else:
            with st.spinner("Analizando tus finanzas..."):
                # 3. Consultar a la IA
                respuesta = consultar_chat_financiero(prompt_usuario, pd.DataFrame(st.session_state['gastos']))
        
        # 4. Guardar respuesta del asistente y recargar para mostrarla
        st.session_state['chat_history'].append({"role": "assistant", "content": respuesta})
        st.rerun()