# ==============================================
# 💬 Chatbot Regulatorio Interno
# Versión completa (Excel + OpenAI + Streamlit)
# ==============================================

import os
from openai import OpenAI
import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import streamlit as st

# ==============================================
# 1️⃣ CONFIGURACIÓN INICIAL
# ==============================================

# Usa tu clave de OpenAI (debe estar guardada en una variable de entorno)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Descargar stopwords si no están instaladas
nltk.download("stopwords", quiet=True)
stop_words = stopwords.words("spanish")

# --- 1️⃣ Cargar el dataset ---
ruta_excel = "conversaciones_revisando.xlsx"
df = pd.read_excel(ruta_excel)

# Normalizar nombres
df.columns = df.columns.str.strip().str.lower()

# Tomar solo las consultas y respuestas
consultas = df[df["role"].str.lower() == "user"]["content"].tolist()
respuestas = df[df["role"].str.lower() == "assistant"]["content"].tolist()

# Asegurar que haya pares del mismo tamaño
pares = list(zip(consultas, respuestas))

# ==============================================
# 2️⃣ CARGAR EMBEDDINGS PRECALCULADOS
# ==============================================
try:
    emb_consultas = np.load("emb_consultas_comprimido.npz")["emb"]
    print("✅ Embeddings cargados correctamente.")
except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'emb_consultas.npy'. Genera primero los embeddings con 'generar_embeddings_excel.py'.")
    st.stop()

# ==============================================
# 4️⃣ Buscar contexto relevante con embeddings
# ==============================================
def buscar_contexto(pregunta, top_k=5):
    """Busca las respuestas más similares semánticamente."""
    emb_pregunta = client.embeddings.create(
        model="text-embedding-3-small", input=pregunta
    ).data[0].embedding
    similitudes = cosine_similarity([emb_pregunta], emb_consultas)[0]
    indices = similitudes.argsort()[-top_k:][::-1]
    fragmentos = [pares[i][1] for i in indices]
    return fragmentos

FRASES_POR_TEMA = {
    "cosmético": [
        "“Un producto cosmético, según el Reglamento (CE) nº 1223/2009, es toda sustancia o mezcla destinada a ser puesta en contacto con las partes superficiales del cuerpo humano (epidermis, sistema piloso y capilar, uñas, labios, órganos genitales externos) o con los dientes y mucosas bucales, con el fin exclusivo o principal de limpiarlos, perfumarlos, modificar su aspecto, protegerlos, mantenerlos en buen estado o corregir los olores corporales.”"
    ],
    "cosmética para animales" : [
        """Los productos destinados a la higiene o cuidado de animales no se consideran cosméticos y quedan fuera del ámbito de aplicación del Reglamento 1223/2009."""
            
        """En el contexto español, estos productos fueron considerados inicialmente como productos zoosanitarios. Tras la publicación del Real Decreto 867/2020 dejaron de estar incluidos en dicho marco, aunque una sentencia del Tribunal Supremo en 2023 anuló parcialmente ese Real Decreto, devolviendo temporalmente a los productos cosméticos para animales la consideración de zoosanitarios.
 
        Finalmente, con la Ley 1/2025, de 1 de abril, que modifica la Ley 8/2003 de sanidad animal, se elimina la obligatoriedad de registro de los productos de higiene, cuidado y manejo de animales (HCM) y del material y utillaje zoosanitario (MUZ). En consecuencia, estos productos quedan fuera del ámbito competencial del Ministerio de Agricultura y Pesca.

        Ante esta situación, el pasado mes de junio nos pusimos en contacto con ASEMAZ, quienes nos informaron de lo siguiente:Con la publicación de la Ley 1/2025, determinados productos zoosanitarios destinados a higiene, cuidado y manejo de los animales ya no tienen que ser notificados por el titular de los mismos para su comercialización.

        Ahora bien, decimos “determinados” dado que dependiendo del “claim” reivindicado por el producto (biocidas), tendrán las siguientes obligaciones:

        **Registro nacional:**
        - Si se trata de un zoosanitario para uso en entorno ganadero (insecticida, larvicida, desinfectante, etc.), deberá solicitarse su registro ante el MAPA como plaguicida, con los correspondientes ensayos según la eficacia que se quiera defender. Más información: https://www.mapa.gob.es/es/ganaderia/temas/sanidad-animal-higiene-ganadera/Higiene-de-la-produccion-primaria-ganadera/registro-de-productos-zoosanitarios/
        - Si se trata de un plaguicida no agrícola (desinfectante de uso en la industria alimentaria o uso ambiental, rodenticida, etc.), deberá solicitarse su registro ante Sanidad como plaguicida no agrícola. Más información: https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regNacional/requisitos_nacional.htm
        - Si se trata de un biocida tipo 3 (higiene veterinaria con función biocida), es obligatoria la notificación a Sanidad de conformidad con la Disposición Transitoria Segunda del RD 1054/2002 (no requiere ensayos de eficacia). Más información: https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regPlaguicidas/dt2notificanuevo.htm

        En todo caso, para los casos anteriores, una vez que las sustancias activas que formen parte del producto (sustancias biocidas) cuenten con Reglamento de Ejecución para los tipos de productos biocidas que se quieren defender, esos productos deberán solicitar su registro por procedimiento europeo, de conformidad con las exigencias del Reglamento (UE) 528/2012.

        En todo caso, si los productos que se deseen comercializar estén afectados o no por lo indicado anteriormente, son productos químicos peligrosos (mezclas o sustancias) quedarán afectados por la normativa de clasificación y etiquetado de mezclas y sustancias químicas, debiendo estar debidamente etiquetados, contar con ficha de datos de seguridad (FDS) y ser notificados a toxicología a través de un expediente PCN.

        Por tanto, tal y como recomiendan desde ASEMAZ, lo más conveniente es poneros en contacto con la autoridad competente correspondiente para que os puedan dar información detallada.
        """
    ]
}

# --- 5️⃣ Generar respuesta con GPT ---
def responder_chatbot(pregunta, mostrar_contexto=False):
    from datetime import datetime

    # --- 🕒 Determinar saludo según hora ---
    hora = datetime.now().hour
    if hora < 12:
        saludo = "Buenos días,"
    else:
        saludo = "Buenas tardes,"

    # --- 🧾 Despedida fija ---
    despedida = (
        "Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.\n"
        "Reciba un cordial saludo,\n"
        "Departamento Técnico."
    )

    pregunta_lower = pregunta.lower()
    fragmentos = buscar_contexto(pregunta)
    if not fragmentos:
        return "No encontré información relevante en la base de datos. ¿Podrías reformular la pregunta?"

    contexto = "\n\n".join(fragmentos)

    # --- Detectar tema ---
    frases_relevantes = []
    for tema, frases in FRASES_POR_TEMA.items():
        if tema in pregunta_lower:
            frases_relevantes.extend(frases)
    
        # --- 🧩 Filtrar la definición general cuando no aporta valor ---
    ingredientes = [
        "formaldehido", "formaldehído", "fenoxietanol", "metanol", "retinol",
        "plomo", "parabenos", "filtros uv", "filtro uv", "perfume",
        "fragancia", "conservante", "colorante", "nanomaterial", "biocida"
    ]

    # Si la pregunta se refiere a un ingrediente, eliminar la definición general de cosmético
    if any(i in pregunta_lower for i in ingredientes):
        if "cosmético" in FRASES_POR_TEMA:
            frases_relevantes = [
                f for f in frases_relevantes
                if f not in FRASES_POR_TEMA["cosmético"]
            ]

    # --- 💬 Construir el prompt técnico con afirmación inicial ---
    frases_texto = "\n".join([f"- {f}" for f in frases_relevantes]) if frases_relevantes else ""

    prompt = f"""
Eres un asistente experto en legislación cosmética, biocidas y productos regulados.

Debes redactar una respuesta **formal, precisa y actualizada**, en tono técnico.
Estructura la respuesta de la siguiente forma:

1️⃣ Comienza con una **afirmación clara y objetiva** sobre la situación normativa del tema preguntado.
   (Ejemplo: “El uso de formaldehído en productos cosméticos comercializados en la Unión Europea está regulado por el Reglamento (CE) nº 1223/2009.”)
2️⃣ Desarrolla a continuación una explicación completa con el contexto legal y técnico.
3️⃣ Finaliza con la despedida establecida.

La respuesta debe empezar con un saludo (“Buenos días,” / “Buenas tardes,”) y finalizar con:

"Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.
Reciba un cordial saludo,
Departamento Técnico."

⚖️ Instrucciones:
- No inventes ni reformules información.
- No incluyas recomendaciones ni valoraciones personales.
- **Debes incluir literalmente las siguientes frases normativas**, sin modificarlas ni traducirlas:
{frases_texto}
- Inserta las frases donde encajen naturalmente en el desarrollo.
- El resto del texto debe complementar las frases con explicaciones objetivas y actuales.

---
Contexto normativo (solo para ampliar datos coherentes con las frases anteriores):
{contexto}

---
Pregunta:
{pregunta}

---
Pregunta:
{pregunta}
"""

    # --- 🔗 Llamada al modelo ---
    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    ).choices[0].message.content.strip()

    # --- 🧩 Evitar duplicados ---
    respuesta_limpia = respuesta.strip()

    if not respuesta_limpia.lower().startswith(("buenos días", "buenas tardes", "buenas noches")):
        respuesta_limpia = f"{saludo}\n\n{respuesta_limpia}"

    despedida_normalizada = despedida.lower().replace("\n", " ").replace("  ", " ").strip()
    respuesta_normalizada = respuesta_limpia.lower().replace("\n", " ").replace("  ", " ").strip()
    if "departamento técnico" not in respuesta_normalizada:
        respuesta_limpia = f"{respuesta_limpia}\n\n{despedida}"

    # --- ✨ Poner en cursiva las frases normativas incluidas ---
    for tema, frases in FRASES_POR_TEMA.items():
        for frase in frases:
            frase_limpia = frase.strip("“”\"'")
            if frase_limpia in respuesta_limpia:
                respuesta_limpia = respuesta_limpia.replace(frase_limpia, f"*{frase_limpia}*")

    return respuesta_limpia

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
