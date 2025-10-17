# ================================================================
# 💬 Chatbot Regulatorio Interno (Excel + OpenAI)
# ================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import openai

# ---------------- CONFIGURACIÓN ----------------
openai.api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Chatbot Regulatorio Interno", page_icon="💬")

EMBED_MODEL = "text-embedding-3-small"
SIM_THRESHOLD = 0.23
TOP_K = 5

# ---------------- CARGA DEL EXCEL ----------------
@st.cache_data(show_spinner=True)
def cargar_datos():
    try:
        df = pd.read_excel("conversaciones_revisando.xlsx")
        df.columns = df.columns.str.strip().str.capitalize()
        df["Role"] = df["Role"].str.lower()
        pares = []
        for i in range(len(df) - 1):
            if df.iloc[i]["Role"] == "user" and df.iloc[i + 1]["Role"] == "assistant":
                consulta = str(df.iloc[i]["Content"]).strip()
                respuesta = str(df.iloc[i + 1]["Content"]).strip()
                pares.append((consulta, respuesta))
        return pares
    except Exception as e:
        st.error(f"⚠️ Error al cargar el Excel: {e}")
        return []

pares = cargar_datos()
if not pares:
    st.stop()

consultas = [c for c, _ in pares]

# ---------------- EMBEDDINGS ----------------
@st.cache_resource(show_spinner=True)
def generar_embeddings(textos):
    res = openai.embeddings.create(model=EMBED_MODEL, input=textos)
    return np.array([d.embedding for d in res.data])

consultas_emb = generar_embeddings(consultas)

# ---------------- FUNCIÓN DE BÚSQUEDA ----------------
def buscar_contexto(pregunta, top_k=TOP_K):
    emb_preg = openai.embeddings.create(model=EMBED_MODEL, input=[pregunta]).data[0].embedding
    emb_preg = np.array(emb_preg)

    similitudes = consultas_emb @ emb_preg / (
        np.linalg.norm(consultas_emb, axis=1) * np.linalg.norm(emb_preg) + 1e-10
    )

    orden = np.argsort(similitudes)[::-1]
    mejores = [pares[i] for i in orden[:top_k] if similitudes[i] >= SIM_THRESHOLD]

    if not mejores:
        return None

    contexto = "\n\n".join(
        [f"Consulta previa: {q}\nRespuesta asociada: {r}" for q, r in mejores]
    )
    return contexto

# ---------------- GENERAR RESPUESTA ----------------
def responder_chatbot(pregunta, frases_forzadas=None):
    contexto = buscar_contexto(pregunta)

    if not contexto:
        return "No tengo información interna suficientemente parecida para responder con seguridad."

    # Añadir frases forzadas (si se especifican)
    citas_extra = ""
    if frases_forzadas:
        citas_extra = "\n\nCitas relevantes (de la base interna):\n" + "\n".join(
            [f"*{f}*" for f in frases_forzadas]
        )

    prompt = f"""
Eres un asistente técnico especializado en regulación cosmética y química.
Responde de manera objetiva, precisa y profesional, basándote únicamente en el siguiente contexto interno y las citas proporcionadas.
No des opiniones ni recomendaciones; explica solo lo que se desprende de la información.
Si la base no contiene suficiente información, indícalo claramente.

Contexto interno:
{contexto}
{citas_extra}

Consulta:
{pregunta}
"""

    respuesta = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    return respuesta.choices[0].message.content.strip()

# ---------------- INTERFAZ STREAMLIT ----------------
st.title("💬 Chatbot Regulatorio Interno")
st.markdown("Asistente basado en tu base interna de consultas y OpenAI GPT-4.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

pregunta = st.chat_input("Escribe tu consulta...")

if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Ejemplo: forzar frases concretas para ciertos temas
    frases_forzadas = None
    if "animal" in pregunta.lower():
        frases_forzadas = [
            """Si se trata de un zoosanitario para uso en entorno ganadero (insecticida, larvicida, desinfectante, etc..) tendrá que solicitarse su registro ante el MAPA como plaguicida, con sus correspondientes ensayos según la eficacia que se quiera defender. La página del MAPA donde podéis informaros es:
https://www.mapa.gob.es/es/ganaderia/temas/sanidad-animal-higiene-ganadera/Higiene-de-la-produccion…
Si se trata de un plaguicida no agrícola (desinfectante de uso en la industria alimentaria o uso ambiental, rodenticida, etc..) tendrá que solicitarse su registro ante el SANIDAD como plaguicida no agrícola, con sus correspondientes ensayos según la eficacia que se quiera defender. La página de SANIDAD donde podéis informaros es:
https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regNacional/requisitos_nacional…
Si se trata de un biocida tipo 3 (higiene veterinaria, pero con función biocida), es obligatorio hacer una notificación a Sanidad de conformidad con la Disposición Transitoria Segunda del RD 1054/2002 (no requiere ensayos de eficacia). El enlace de Sanidad donde encontráis la información es el siguiente:
https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regPlaguicidas/dt2notificanuevo…
En todo caso, para los casos anteriores, una vez que las sustancias activas que formen parte del producto (sustancias biocidas) cuenten con Reglamento de Ejecución para los tipos de productos biocidas que se quieren defender, esos productos deberán solicitar su registro por procedimiento europeo, de conformidad con las exigencias del Reglamento (UE) 528/2012.
En todo caso, si los productos que se deseen comercializar estén afectados o no por lo indicado anteriormente, son productos químicos peligrosos (mezclas o sustancias) quedarán afectados por la normativa de clasificación y etiquetado de mezclas y sustancias químicas, debiendo estar debidamente etiquetados, contar con ficha de datos de seguridad (FDS) y ser notificados a toxicología a través de un expediente PCN.
No obstante, consideramos que lo conveniente es que vuestra empresa realice la consulta a la autoridad competente correspondiente, para que proporcionen una opinión fundada sobre los productos que desean fabricar/comercializar."""
        ]

    respuesta = responder_chatbot(pregunta, frases_forzadas)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})

    with st.chat_message("assistant"):
        st.markdown(respuesta)