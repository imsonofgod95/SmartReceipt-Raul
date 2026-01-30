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
# 1. CONFIGURACIÓN SEGURA (CLOUD READY)
# =======================================================
st.set_page_config(page_title="SmartReceipt Cloud", layout="wide", page_icon="☁️")

try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("⚠️ Falta la API Key en Secrets")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error API: {e}")
    st.stop()

ARCHIVO_DB = "historial_gastos.csv"

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
# 2. CARGA INICIAL
# =======================================================
if "gastos" not in st.session_state:
    if os.path.exists(ARCHIVO_DB):
        try:
            df = pd.read_csv(ARCHIVO_DB)
            for c in ["lat", "lon"]:
                if c not in df.columns:
                    df[c] = 0.0
            st.session_state["gastos"] = df.to_dict("records")
        except:
            st.session_state["gastos"] = []
    else:
        st.session_state["gastos"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# =======================================================
# 3. PROCESAMIENTO VISUAL (SOLO PARA MOSTRAR)
# =======================================================
def procesar_imagen_opencv(imagen_pil):
    img = np.array(imagen_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    return Image.fromarray(thresh)

# =======================================================
# 4. IA – MODELO
# =======================================================
def obtener_modelo_valido():
    try:
        modelos = [m.name for m in genai.list_models()
                   if "generateContent" in m.supported_generation_methods]
        for m in modelos:
            if "flash" in m and "1.5" in m:
                return m
        return modelos[0]
    except:
        return None

# =======================================================
# 5. ANALIZAR TICKET (VISION)
# =======================================================
def analizar_ticket(imagen_original):
    modelo = obtener_modelo_valido()
    if not modelo:
        return "{}", ""

    prompt = f"""
Analiza este ticket (puede ser Costco, Juguetron u otro).
Devuelve SOLO JSON válido.

REGLAS IMPORTANTES:
- Costco suele poner TOTAL al final
- Fecha puede venir como DD/MM/AAAA o DD-MM-AAAA
- Ubicación = sucursal (NO razón social)
- Si no hay algo claro, infiere lo más probable

JSON:
{{
  "comercio": "Texto",
  "total": 0.00,
  "fecha": "DD/MM/AAAA",
  "ubicacion": "Sucursal",
  "latitud": 19.4326,
  "longitud": -99.1332,
  "categoria": "Varios",
  "detalles": "Texto libre"
}}
"""
    try:
        model = genai.GenerativeModel(modelo)
        response = model.generate_content([prompt, imagen_original])
        return response.text, modelo
    except Exception as e:
        return f"{{}}", modelo

# =======================================================
# 6. CHAT FINANCIERO
# =======================================================
def consultar_chat_financiero(pregunta, df):
    modelo = obtener_modelo_valido()
    model = genai.GenerativeModel(modelo)
    csv = df.to_csv(index=False)

    prompt = f"""
Eres un asistente financiero.
Datos CSV:
{csv}

Pregunta:
{pregunta}
"""
    try:
        r = model.generate_content(prompt)
        return r.text
    except:
        return "Error analizando datos."

# =======================================================
# 7. UI
# =======================================================
st.title("🧠 SmartReceipt – Cloud Edition")
tab1, tab2, tab3 = st.tabs(["📸 Nuevo Ticket", "📊 Dashboard", "💬 Chat IA"])

# ===================== TAB 1 ===========================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        archivo = st.file_uploader("Sube ticket", type=["jpg", "jpeg", "png"])
        if archivo:
            img_original = Image.open(archivo).convert("RGB")
            img_proc = procesar_imagen_opencv(img_original)

            st.image(img_proc, caption="Procesada (solo visual)", use_container_width=True)

            if st.button("🧠 Escanear", type="primary"):
                with st.spinner("Analizando ticket..."):
                    texto, _ = analizar_ticket(img_original)

                    limpio = texto.replace("```json", "").replace("```", "").strip()
                    if "{" in limpio:
                        limpio = limpio[limpio.find("{"): limpio.rfind("}") + 1]

                    try:
                        data = json.loads(limpio)

                        # Normalizaciones defensivas
                        data["total"] = float(str(data.get("total", 0)).replace("$", "").replace(",", ""))
                        if not re.match(r"\d{2}/\d{2}/\d{4}", data.get("fecha", "")):
                            data["fecha"] = ""

                        st.session_state["temp_data"] = data
                        st.toast("Ticket leído correctamente", icon="✅")
                    except:
                        st.error("No se pudo interpretar el ticket")

    with col2:
        if "temp_data" in st.session_state:
            d = st.session_state["temp_data"]
            with st.form("guardar"):
                st.subheader("Validar datos")

                vc = st.text_input("Comercio", d.get("comercio", ""))
                vm = st.number_input("Monto", value=d.get("total", 0.0))
                vf = st.text_input("Fecha", d.get("fecha", ""))

                cat = d.get("categoria", "Varios")
                idx = LISTA_CATEGORIAS.index(cat) if cat in LISTA_CATEGORIAS else -1
                vcat = st.selectbox("Categoría", LISTA_CATEGORIAS, index=idx)

                vu = st.text_input("Sucursal", d.get("ubicacion", ""))
                vlat = st.number_input("Lat", value=float(d.get("latitud", 19.4326)))
                vlon = st.number_input("Lon", value=float(d.get("longitud", -99.1332)))
                vdet = st.text_input("Detalles", d.get("detalles", ""))

                if st.form_submit_button("💾 Guardar"):
                    st.session_state["gastos"].append({
                        "Fecha": vf,
                        "Comercio": vc,
                        "Monto": vm,
                        "Ubicación": vu,
                        "lat": vlat,
                        "lon": vlon,
                        "Categoría": vcat,
                        "Detalles": vdet
                    })
                    pd.DataFrame(st.session_state["gastos"]).to_csv(ARCHIVO_DB, index=False)
                    del st.session_state["temp_data"]
                    st.success("Guardado")
                    st.rerun()

# ===================== TAB 2 ===========================
with tab2:
    if st.session_state["gastos"]:
        df = pd.DataFrame(st.session_state["gastos"])
        st.metric("💰 Total", f"${df['Monto'].sum():,.2f}")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Aún no hay datos")

# ===================== TAB 3 ===========================
with tab3:
    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Pregunta sobre tus gastos...")
    if q:
        st.session_state["chat_history"].append({"role": "user", "content": q})
        with st.chat_message("assistant"):
            r = consultar_chat_financiero(q, pd.DataFrame(st.session_state["gastos"]))
            st.markdown(r)
        st.session_state["chat_history"].append({"role": "assistant", "content": r})
