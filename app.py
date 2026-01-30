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
import hashlib

# =======================================================
# 1. CONFIGURACIÓN SEGURA ☁️
# =======================================================
st.set_page_config(page_title="SmartReceipt Cloud", layout="wide", page_icon="☁️")

try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("⚠️ Falta la API Key en Secrets.")
        st.stop()
except Exception as e:
    st.error(f"Error de configuración: {e}")
    st.stop()

ARCHIVO_DB = "historial_gastos.csv"
CACHE_FILE = "cache_ia.json"

# =======================================================
# 2. CATÁLOGOS MAESTROS
# =======================================================
LISTA_CATEGORIAS = [
    "Alimentos y Supermercado", "Restaurantes y Bares", "Gasolina y Transporte",
    "Salud y Farmacia", "Hogar y Muebles", "Servicios (Luz/Agua/Internet)",
    "Telefonía y Comunicaciones", "Ropa y Calzado", "Electrónica y Tecnología",
    "Entretenimiento y Cine", "Educación y Libros", "Mascotas",
    "Regalos y Detalles", "Viajes y Hoteles", "Suscripciones (Streaming)",
    "Cuidado Personal y Belleza", "Deportes y Gimnasio", "Oficina y Trabajo",
    "Mantenimiento Automotriz", "Varios"
]

COMERCIOS_CANONICOS = {
    "COSTCO": ["COSTCO", "COSTCO WHOLESALE", "COSTCO GAS"],
    "JUGUETRON": ["JUGUETRON", "JUGUETRÓN"],
    "WALMART": ["WALMART", "WAL MART"],
    "OXXO": ["OXXO"]
}

def normalizar_comercio(nombre):
    nombre = nombre.upper()
    for canonico, variantes in COMERCIOS_CANONICOS.items():
        for v in variantes:
            if v in nombre:
                return canonico
    return nombre

# =======================================================
# 3. CACHE IA (ANTI CUOTA / ANTI REPETICIÓN)
# =======================================================
def hash_imagen(imagen):
    return hashlib.md5(imagen.tobytes()).hexdigest()

def cargar_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def guardar_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# =======================================================
# 4. CARGA BASE DE DATOS
# =======================================================
if "gastos" not in st.session_state:
    if os.path.exists(ARCHIVO_DB):
        try:
            df = pd.read_csv(ARCHIVO_DB)
            if "lat" not in df: df["lat"] = 0.0
            if "lon" not in df: df["lon"] = 0.0
            st.session_state["gastos"] = df.to_dict("records")
        except:
            st.session_state["gastos"] = []
    else:
        st.session_state["gastos"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# =======================================================
# 5. VISIÓN POR COMPUTADORA (SUAVE)
# =======================================================
def procesar_imagen_opencv(imagen_pil):
    img_np = np.array(imagen_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)

# =======================================================
# 6. MODELO IA ROBUSTO
# =======================================================
def obtener_modelo_valido():
    try:
        modelos = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for m in modelos:
            if "gemini-1.5-flash" in m:
                return m
        return modelos[0]
    except:
        return "gemini-1.5-flash"

def analizar_ticket(imagen):
    modelo = obtener_modelo_valido()
    model = genai.GenerativeModel(modelo)
    cats = ", ".join(LISTA_CATEGORIAS)

    prompt = f"""
Analiza esta imagen de un ticket REAL.

REGLAS:
- No inventes datos.
- Si no existe un campo, usa null.
- El total es el MONTO FINAL PAGADO.
- Usa DD/MM/AAAA.
- Comercio corto y reconocible.

Categorías válidas: [{cats}]

Devuelve SOLO JSON:

{{
  "comercio": "",
  "total": null,
  "fecha": null,
  "ubicacion": null,
  "latitud": null,
  "longitud": null,
  "categoria": "",
  "detalles": ""
}}
"""
    response = model.generate_content([prompt, imagen])
    return response.text

# =======================================================
# 7. UI PRINCIPAL
# =======================================================
st.title("🧠 SmartReceipt – Business Ready")

tab1, tab2, tab3 = st.tabs(["📸 Nuevo Ticket", "📊 Dashboard", "💬 Chat IA"])

# -------------------------------------------------------
# TAB 1 – NUEVO TICKET
# -------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        archivo = st.file_uploader("Sube tu ticket", type=["jpg","png","jpeg"])
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen_opencv(img)
            st.image(img_proc, use_container_width=True)

            if st.button("🧠 Escanear"):
                cache = cargar_cache()
                h = hash_imagen(img)

                if h in cache:
                    st.session_state["temp"] = cache[h]
                    st.toast("Leído desde caché ⚡")
                else:
                    with st.spinner("Analizando con IA..."):
                        raw = analizar_ticket(img)
                        clean = raw.replace("```json","").replace("```","").strip()
                        clean = clean[clean.find("{"):clean.rfind("}")+1]
                        data = json.loads(clean)
                        cache[h] = data
                        guardar_cache(cache)
                        st.session_state["temp"] = data

    with col2:
        if "temp" in st.session_state:
            d = st.session_state["temp"]

            vc = normalizar_comercio(d.get("comercio",""))
            vm = d.get("total") or 0.0
            vf = d.get("fecha") or pd.Timestamp.today().strftime("%d/%m/%Y")
            vu = d.get("ubicacion") or vc

            try: vm = float(vm)
            except: vm = 0.0

            with st.form("save"):
                vc = st.text_input("Comercio", vc)
                vm = st.number_input("Monto", value=float(vm))
                vf = st.text_input("Fecha", vf)
                vcat = st.selectbox("Categoría", LISTA_CATEGORIAS)
                vu = st.text_input("Ubicación", vu)
                vdet = st.text_input("Detalles", d.get("detalles",""))

                if st.form_submit_button("💾 Guardar"):
                    st.session_state["gastos"].append({
                        "Fecha": vf,
                        "Comercio": vc,
                        "Monto": vm,
                        "Ubicación": vu,
                        "lat": d.get("latitud",0),
                        "lon": d.get("longitud",0),
                        "Categoría": vcat,
                        "Detalles": vdet
                    })
                    pd.DataFrame(st.session_state["gastos"]).to_csv(ARCHIVO_DB, index=False)
                    del st.session_state["temp"]
                    st.success("Guardado ✔")
                    st.rerun()

# -------------------------------------------------------
# TAB 2 – DASHBOARD
# -------------------------------------------------------
with tab2:
    if st.session_state["gastos"]:
        df = pd.DataFrame(st.session_state["gastos"])
        st.metric("💰 Total", f"${df['Monto'].sum():,.2f}")
        st.metric("🧾 Tickets", len(df))
        st.altair_chart(
            alt.Chart(df).mark_arc(innerRadius=50).encode(
                theta="Monto", color="Categoría"
            ), use_container_width=True
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Sin datos aún")

# -------------------------------------------------------
# TAB 3 – CHAT IA
# -------------------------------------------------------
with tab3:
    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Pregunta sobre tus gastos")
    if q:
        st.session_state["chat_history"].append({"role":"user","content":q})
        with st.chat_message("assistant"):
            if not st.session_state["gastos"]:
                ans = "No hay datos todavía."
            else:
                model = genai.GenerativeModel(obtener_modelo_valido())
                csv = pd.DataFrame(st.session_state["gastos"]).to_csv(index=False)
                ans = model.generate_content(
                    f"Analiza estos datos:\n{csv}\nPregunta:{q}"
                ).text
            st.markdown(ans)
        st.session_state["chat_history"].append({"role":"assistant","content":ans})
