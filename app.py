import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
import os
import altair as alt
import pydeck as pdk
import hashlib
import time

# =======================================================
# 1. CONFIGURACIÓN
# =======================================================
st.set_page_config(page_title="SmartReceipt Business", layout="wide", page_icon="🧾")

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except:
    st.error("❌ GOOGLE_API_KEY no configurada en Secrets")
    st.stop()

ARCHIVO_DB = "historial_gastos.csv"
CACHE_FILE = "cache_ia.json"

# =======================================================
# 2. CATÁLOGOS
# =======================================================
LISTA_CATEGORIAS = [
    "Alimentos y Supermercado", "Restaurantes y Bares",
    "Gasolina y Transporte", "Salud y Farmacia",
    "Hogar y Muebles", "Servicios",
    "Ropa y Calzado", "Entretenimiento",
    "Electrónica", "Viajes", "Varios"
]

COMERCIOS_CANONICOS = {
    "COSTCO": ["COSTCO"],
    "JUGUETRON": ["JUGUETRON", "JUGUETRÓN"],
    "WALMART": ["WALMART"],
    "OXXO": ["OXXO"]
}

def normalizar_comercio(nombre):
    nombre = (nombre or "").upper()
    for canonico, variantes in COMERCIOS_CANONICOS.items():
        for v in variantes:
            if v in nombre:
                return canonico
    return nombre or "NO DETECTADO"

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
# 4. CARGA BD
# =======================================================
if "gastos" not in st.session_state:
    if os.path.exists(ARCHIVO_DB):
        st.session_state["gastos"] = pd.read_csv(ARCHIVO_DB).to_dict("records")
    else:
        st.session_state["gastos"] = []

# =======================================================
# 5. PREPROCESAMIENTO IMAGEN
# =======================================================
def procesar_imagen(imagen):
    img = np.array(imagen)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)

# =======================================================
# 6. IA ROBUSTA CON FALLBACK
# =======================================================
def obtener_modelo():
    return "gemini-1.5-flash"

def analizar_ticket(imagen):
    model = genai.GenerativeModel(obtener_modelo())

    prompt = """
Analiza este ticket REAL.

REGLAS:
- NO inventes
- Si no existe → null
- Fecha DD/MM/AAAA
- Total = monto final pagado

Devuelve SOLO JSON:

{
 "comercio": null,
 "total": null,
 "fecha": null,
 "ubicacion": null,
 "latitud": null,
 "longitud": null,
 "categoria": null,
 "detalles": null
}
"""

    try:
        response = model.generate_content([prompt, imagen])
        texto = response.text.replace("```json","").replace("```","").strip()
        texto = texto[texto.find("{"):texto.rfind("}")+1]
        return json.loads(texto)

    except Exception as e:
        # 🔐 FALLBACK SEGURO
        st.warning("⚠️ Error IA. Usando fallback seguro.")
        return {
            "comercio": None,
            "total": None,
            "fecha": None,
            "ubicacion": None,
            "latitud": None,
            "longitud": None,
            "categoria": "Varios",
            "detalles": None
        }

# =======================================================
# 7. UI
# =======================================================
st.title("🧠 SmartReceipt – Business Hardened")

tab1, tab2 = st.tabs(["📸 Nuevo Ticket", "📊 Dashboard"])

# ---------------- TAB 1 ----------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        archivo = st.file_uploader("Sube tu ticket", ["jpg","png","jpeg"])
        if archivo:
            img = Image.open(archivo)
            img_proc = procesar_imagen(img)
            st.image(img_proc, use_container_width=True)

            if st.button("🧠 Analizar"):
                cache = cargar_cache()
                h = hash_imagen(img_proc)

                if h in cache:
                    data = cache[h]
                    st.toast("Leído desde caché ⚡")
                else:
                    with st.spinner("Analizando con IA..."):
                        data = analizar_ticket(img_proc)
                        cache[h] = data
                        guardar_cache(cache)
                        time.sleep(1)

                st.session_state["temp"] = data

    with col2:
        if "temp" in st.session_state:
            d = st.session_state["temp"]

            comercio = normalizar_comercio(d.get("comercio"))
            monto = d.get("total") or 0.0
            fecha = d.get("fecha") or pd.Timestamp.today().strftime("%d/%m/%Y")
            ubicacion = d.get("ubicacion") or comercio

            with st.form("guardar"):
                comercio = st.text_input("Comercio", comercio)
                monto = st.number_input("Monto", value=float(monto))
                fecha = st.text_input("Fecha", fecha)
                categoria = st.selectbox("Categoría", LISTA_CATEGORIAS)
                ubicacion = st.text_input("Ubicación", ubicacion)
                detalles = st.text_input("Detalles", d.get("detalles") or "")

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
                    st.success("✅ Guardado")
                    st.rerun()

# ---------------- TAB 2 ----------------
with tab2:
    if st.session_state["gastos"]:
        df = pd.DataFrame(st.session_state["gastos"])
        st.metric("💰 Total", f"${df['Monto'].sum():,.2f}")
        st.metric("🧾 Tickets", len(df))

        st.altair_chart(
            alt.Chart(df).mark_arc(innerRadius=40).encode(
                theta="Monto",
                color="Categoría"
            ),
            use_container_width=True
        )

        st.dataframe(df, use_container_width=True)
    else:
        st.info("Aún no hay tickets")

