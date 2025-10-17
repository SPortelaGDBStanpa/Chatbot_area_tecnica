# ==============================================
# 💬 Chatbot Regulatorio Interno
# Versión completa (Excel + OpenAI + Streamlit)
# ==============================================

import os
import openai
import pandas as pd
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import streamlit as st

# ==============================================
# 1️⃣ CONFIGURACIÓN INICIAL
# ==============================================

# Usa tu clave de OpenAI (debe estar guardada en una variable de entorno)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Descargar stopwords si no están instaladas
nltk.download("stopwords", quiet=True)
stop_words = stopwords.words("spanish")

# ==============================================
# 2️⃣ CARGAR EL DATASET (Excel)
# ==============================================
@st.cache_data(show_spinner=False)
def cargar_datos():
    try:
        df = pd.read_excel("conversaciones_revisando.xlsx")
        df.columns = df.columns.str.strip().str.capitalize()
        df["Role"] = df["Role"].str.lower()
        consultas = df[df["Role"] == "user"]["Content"].fillna("").tolist()
        respuestas = df[df["Role"] == "assistant"]["Content"].fillna("").tolist()
        pares = list(zip(consultas, respuestas))
        return pares
    except Exception as e:
        st.error(f"⚠️ Error al cargar el archivo Excel: {e}")
        return []

pares = cargar_datos()
if not pares:
    st.stop()

st.sidebar.success(f"📄 Dataset cargado con {len(pares)} pares de consulta-respuesta.")

# ==============================================
# 3️⃣ VECTORIZACIÓN DE TEXTO
# ==============================================
@st.cache_resource(show_spinner=False)
def crear_vectorizador():
    vectorizador = TfidfVectorizer(stop_words=stop_words)
    X = vectorizador.fit_transform([c for c, _ in pares])
    return vectorizador, X

vectorizador, X = crear_vectorizador()

# ==============================================
# 4️⃣ FUNCIÓN PRINCIPAL DE RESPUESTA
# ==============================================
def responder_chatbot(pregunta, top_k=3):
    # Buscar las consultas más parecidas
    vector_pregunta = vectorizador.transform([pregunta])
    similitudes = cosine_similarity(vector_pregunta, X).flatten()
    indices_top = similitudes.argsort()[::-1][:top_k]
    
    # Crear el contexto
    contexto = "\n\n".join(
        [f"Consulta: {pares[i][0]}\nRespuesta: {pares[i][1]}" for i in indices_top]
    )

    # Generar el prompt
    prompt = f"""
Eres un asistente experto en regulación cosmética.
Responde de forma clara, técnica y concisa a la siguiente consulta,
basándote en el contexto histórico del equipo técnico.

Contexto relevante:
{contexto}

Nueva consulta:
{pregunta}

Si no hay información suficiente, indica que no dispones de datos internos al respecto.
"""

    # Llamada a la API de OpenAI
    try:
        respuesta = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0,  # garantiza respuestas consistentes
            messages=[{"role": "user", "content": prompt}]
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error al generar la respuesta: {e}"

# ==============================================
# 5️⃣ INTERFAZ STREAMLIT
# ==============================================
st.title("💬 Chatbot Regulatorio Interno")
st.write("Escribe tu consulta relacionada con normativa cosmética o procedimientos técnicos.")

pregunta = st.text_area("🧴 Tu consulta:")
if st.button("Enviar"):
    if pregunta.strip():
        with st.spinner("Analizando consulta..."):
            respuesta = responder_chatbot(pregunta)
        st.markdown(f"### 💬 Respuesta:\n{respuesta}")
    else:
        st.warning("Por favor, escribe una consulta antes de enviar.")

st.markdown("---")
st.caption("🧠 Basado en el histórico de consultas internas y el modelo GPT-4o de OpenAI.")
