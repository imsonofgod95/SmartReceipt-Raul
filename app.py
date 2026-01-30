import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
import os
import altair as alt
import hashlib

# =======================================================
# 1. CONFIGURACIÓN
# =======================================================
st.set_page_config(page_title="SmartReceipt Cloud", layout="wide", page_icon="☁️")

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("⚠️ Falta GOOGLE_API_KEY en Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

ARCHIVO_DB = "historial_gastos.csv"
CACHE_FILE = "cache_ia.json"

# =======================================================
# 2. CATÁLOGOS
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
    "COSTCO": ["COSTCO"],
    "JUGUETRON": ["JUGUETRON", "JUGUETRÓN"],
    "WALMART": ["WALMART"],
    "OXXO": ["OXXO"]
}

def normalizar_comercio(nombre):
    if not nombre:
        return ""
    nombre = nombre.upper()
    for canon, variantes in COMERCIOS_CANONICOS.items():
        for v in variantes:
            if v in nombre:
                return canon
    return nombre

# =======================================================
# 3. CACHE IA
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
# 4. BASE DE DATOS
# =======================================================
if "gastos" not in st.session_state:
    if os.path.exists(ARCHIVO_DB):
        df = pd.read_csv(ARCHIVO_DB)
        st.session_state["gastos"] = df.to_dict("records")
    else:
        st.session_state["gastos"] = []

# =======================================================
# 5. VISIÓN COMPUTACIONAL (LIGERA)
# =======================================================
def procesar_imagen_opencv(imagen):
    img = np.array(imagen)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    img = clahe.apply(img)
    return Image.fromarray(img)

# =======================================================
# 6. MODELO IA ROBUSTO
# =======================================================
def obtener_modelo_valido():
    try:
        modelos = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for m in modelos:
            if "gemini-1.5-flash" in m:
                return m
        return modelos[0]
    except:
        return "gemini-1.5-flash"

def analizar_ticket(imagen):
    modelo = obtener_modelo_valido()
    model = genai.GenerativeModel(modelo)
    categorias = ", ".join(LISTA_CATEGORIAS)

    prompt = f"""
Analiza este ticket.

REGLAS:
- No inventes datos
- Si no existe, usa null
- Fecha DD/MM/AAAA
- Total = monto final pagado

Categorías válidas: [{categorias}]

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

    try:
        response = model.generate_content([prompt, imagen])
        return response.text

    except Exception as e:
        error = str(e)

        if "ResourceExhausted" in error:
            st.warning("⚠️ Cuota de IA agotada. Usando fallback.")
        else:
            st.error("❌ Error IA. Usando fallback seguro.")

        return json.dumps({
            "comercio": None,
            "total": None,
            "fecha": None,
            "ubicacion": None,
            "latitud": None,
            "longitud": None,
            "categoria": "Varios",
            "detalles": "Fallback IA"
        })

# =======================================================
# 7. UI
# =======================================================
st.title("🧠 SmartReceipt – Business Ready")

tab1, tab2 = st.tabs(["📸 Nuevo Ticket", "📊 Dashboard"])

# -------------------------------------------------------
# TAB 1
# -------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        archivo = st.file_uploader("Sube tu ticket", type=["jpg", "png", "jpeg"])

        if archivo:
            img_original = Image.open(archivo).convert("RGB")
            img_proc = procesar_imagen_opencv(img_original)
            st.image(img_proc, caption="Procesada", use_container_width=True)

            if st.button("🧠 Escanear"):
                cache = cargar_cache()
                h = hash_imagen(img_proc)

                if h in cache:
                    st.session_state["temp"] = cache[h]
                    st.toast("⚡ Desde caché")
                else:
                    with st.spinner("Analizando..."):
                        raw = analizar_ticket(img_proc)
                        clean = raw.replace("```json", "").replace("```", "").strip()
                        clean = clean[clean.find("{"):clean.rfind("}") + 1]
                        data = json.loads(clean)

                        cache[h] = data
                        guardar_cache(cache)
                        st.session_state["temp"] = data

    with col2:
        if "temp" in st.session_state:
            d = st.session_state["temp"]

            comercio = normalizar_comercio(d.get("comercio"))
            monto = float(d.get("total") or 0)
            fecha = d.get("fecha") or pd.Timestamp.today().strftime("%d/%m/%Y")
            ubicacion = d.get("ubicacion") or comercio

            with st.form("guardar"):
                comercio = st.text_input("Comercio", comercio)
                monto = st.number_input("Monto", value=monto)
                fecha = st.text_input("Fecha", fecha)
                categoria = st.selectbox("Categoría", LISTA_CATEGORIAS)
                ubicacion = st.text_input("Ubicación", ubicacion)
                detalles = st.text_input("Detalles", d.get("detalles", ""))

                if st.form_submit_button("💾 Guardar"):
                    st.session_state["gastos"].append({
                        "Fecha": fecha,
                        "Comercio": comercio,
                        "Monto": monto,
                        "Ubicación": ubicacion,
                        "Categoría": categoria,
                        "Detalles": detalles
                    })
                    pd.DataFrame(st.session_state["gastos"]).to_csv(ARCHIVO_DB, index=False)
                    del st.session_state["temp"]
                    st.success("Guardado ✔")
                    st.rerun()

# -------------------------------------------------------
# TAB 2
# -------------------------------------------------------
with tab2:
    if st.session_state["gastos"]:
        df = pd.DataFrame(st.session_state["gastos"])
        st.metric("💰 Total", f"${df['Monto'].sum():,.2f}")
        st.metric("🧾 Tickets", len(df))
        st.altair_chart(
            alt.Chart(df).mark_arc(innerRadius=50).encode(
                theta="Monto", color="Categoría"
            ),
            use_container_width=True
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Sin datos aún")

