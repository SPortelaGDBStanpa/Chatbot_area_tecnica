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
import unicodedata
import markdown
import re

def render_html_markdown(texto):
    """Convierte markdown a HTML dentro del contenedor estilizado."""
    html = markdown.markdown(texto, extensions=["extra", "sane_lists"])
    return f"<div class='chat-response'>{html}</div>"

def quitar_acentos(texto):
    """Elimina acentos y caracteres diacríticos de un texto."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def detectar_redireccion(fragmentos):
    """Detecta si alguna respuesta del contexto indica redirección a otro departamento."""
    palabras_clave = [
        "contacte con", "diríjase a", "debe consultarlo con", "responsabilidad de",
        "departamento de", "servicio de", "autoridad competente", "remítase a",
        "redirigirse a", "consultar con", "trasladar la consulta a"
    ]
    texto_unido = " ".join(fragmentos).lower()
    return any(p in texto_unido for p in palabras_clave)

def extraer_redireccion(fragmentos):
    """Extrae el fragmento que contiene una instrucción de redirección e identifica el departamento o responsable."""
    palabras_clave = [
        "contacte con", "diríjase a", "debe consultarlo con", "responsabilidad de",
        "departamento de", "servicio de", "autoridad competente", "remítase a",
        "redirigirse a", "consultar con", "trasladar la consulta a", "deberás escribir un correo",
        "deberás enviar un correo", "correo a la dirección", "puedes contactar con",
        "escriba a", "escribir a", "envíe un correo", "mandar un correo"
    ]

    departamentos = [
        "internacional", "reglamentación", "toxicología", "formulación",
        "medio ambiente", "jurídico", "seguridad del producto",
        "autoridad competente", "asuntos científicos", "asuntos regulatorios"
    ]

    for frag in fragmentos:
        texto = frag.lower()
        if any(p in texto for p in palabras_clave):
            # Buscar si se menciona un departamento específico
            for palabra in departamentos:
                if palabra in texto:
                    return f"{frag.strip()} (Corresponde al departamento de {palabra.capitalize()})."

            # Buscar si hay dirección de correo
            if "@" in frag:
                return f"{frag.strip()} (Se indica una dirección de correo para contacto directo)."

            return frag.strip()

    return None

# ==============================================
# 1️⃣ CONFIGURACIÓN INICIAL
# ==============================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
nltk.download("stopwords", quiet=True)
stop_words = stopwords.words("spanish")

ruta_excel = "conversaciones_revisando.xlsx"
df = pd.read_excel(ruta_excel)
df.columns = df.columns.str.strip().str.lower()

consultas = df[df["role"].str.lower() == "user"]["content"].tolist()
respuestas = df[df["role"].str.lower() == "assistant"]["content"].tolist()
pares = list(zip(consultas, respuestas))

# ==============================================
# 2️⃣ CARGAR EMBEDDINGS PRECALCULADOS
# ==============================================
try:
    emb_consultas = np.load("emb_consultas_comprimido.npz")["emb"]
    print("✅ Embeddings cargados correctamente.")
except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'emb_consultas_comprimido.npz'.")
    st.stop()

# ==============================================
# 4️⃣ Buscar contexto relevante con embeddings
# ==============================================
def buscar_contexto(pregunta, top_k=5, umbral_similitud=0.65):
    """
    Busca los fragmentos más relevantes del Excel usando embeddings.
    Si encuentra una coincidencia casi exacta (similitud >= 0.80), devuelve esa respuesta literal.
    """
    pregunta_sin_acentos = quitar_acentos(pregunta.lower())

    # Crear embedding de la pregunta
    emb_pregunta = client.embeddings.create(
        model="text-embedding-3-small",
        input=pregunta_sin_acentos
    ).data[0].embedding

    # Calcular similitudes con las consultas del Excel
    similitudes = cosine_similarity([emb_pregunta], emb_consultas)[0]
    indices_ordenados = similitudes.argsort()[::-1]  # de mayor a menor

    # 🧩 Mostrar las coincidencias más altas en consola (para diagnóstico)
    print("\n--- 🔍 Diagnóstico de similitudes ---")
    for i in indices_ordenados[:5]:
        print(f"{similitudes[i]:.3f} → {pares[i][0][:90]}...")
    print("--------------------------------------\n")

    # ✅ NUEVO: si hay una coincidencia muy fuerte, usar la respuesta literal del Excel
    max_sim = similitudes[indices_ordenados[0]]
    if max_sim >= 0.80:
        print(f"✅ Coincidencia fuerte detectada ({max_sim:.3f}) — usando respuesta literal del Excel.")
        return [pares[indices_ordenados[0]][1]]

    # 🔹 Si no hay coincidencias fuertes, usar los fragmentos más similares
    indices_validos = [i for i, s in enumerate(similitudes) if s >= umbral_similitud]
    if not indices_validos:
        indices_validos = similitudes.argsort()[-2:][::-1]

    indices_ordenados = sorted(indices_validos, key=lambda i: similitudes[i], reverse=True)[:top_k]
    fragmentos = [pares[i][1] for i in indices_ordenados]

    # 🧹 Eliminar duplicados
    fragmentos = list(dict.fromkeys(fragmentos))
    return fragmentos

# ==============================================
# FRASES POR TEMA
# ==============================================
FRASES_POR_TEMA = {
    "cosmetico": [
        "“Un producto cosmético, según el Reglamento (CE) nº 1223/2009, es toda sustancia o mezcla destinada a ser puesta en contacto con las partes superficiales del cuerpo humano (epidermis, sistema piloso y capilar, uñas, labios, órganos genitales externos) o con los dientes y mucosas bucales, con el fin exclusivo o principal de limpiarlos, perfumarlos, modificar su aspecto, protegerlos, mantenerlos en buen estado o corregir los olores corporales.”"
    ],
    "cosmetica para animales" : [
        """Los productos destinados a la higiene o cuidado de animales no se consideran cosméticos y quedan fuera del ámbito de aplicación del Reglamento 1223/2009.
            
En el contexto español, estos productos fueron considerados inicialmente como productos zoosanitarios. Tras la publicación del Real Decreto 867/2020 dejaron de estar incluidos en dicho marco, aunque una sentencia del Tribunal Supremo en 2023 anuló parcialmente ese Real Decreto, devolviendo temporalmente a los productos cosméticos para animales la consideración de zoosanitarios.

Finalmente, con la Ley 1/2025, de 1 de abril, que modifica la Ley 8/2003 de sanidad animal, se elimina la obligatoriedad de registro de los productos de higiene, cuidado y manejo de animales (HCM) y del material y utillaje zoosanitario (MUZ). En consecuencia, estos productos quedan fuera del ámbito competencial del Ministerio de Agricultura y Pesca.

Ante esta situación, el pasado mes de junio nos pusimos en contacto con ASEMAZ, quienes nos informaron de lo siguiente:Con la publicación de la Ley 1/2025, determinados productos zoosanitarios destinados a higiene, cuidado y manejo de los animales ya no tienen que ser notificados por el titular de los mismos para su comercialización.

Ahora bien, decimos “determinados” dado que dependiendo del “claim” reivindicado por el producto (biocidas), tendrán las siguientes obligaciones:

**Registro nacional:**\n
- Si se trata de un zoosanitario para uso en entorno ganadero (insecticida, larvicida, desinfectante, etc.), deberá solicitarse su registro ante el **MAPA** como plaguicida, con los correspondientes ensayos según la eficacia que se quiera defender.  
  Más información: [Registro de productos zoosanitarios - MAPA](https://www.mapa.gob.es/es/ganaderia/temas/sanidad-animal-higiene-ganadera/Higiene-de-la-produccion-primaria-ganadera/registro-de-productos-zoosanitarios/)\n
- Si se trata de un plaguicida no agrícola (desinfectante de uso en la industria alimentaria o uso ambiental, rodenticida, etc.), deberá solicitarse su registro ante **Sanidad** como plaguicida no agrícola.  
  Más información: [Registro nacional de plaguicidas no agrícolas - Ministerio de Sanidad](https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regNacional/requisitos_nacional.htm)\n
- Si se trata de un **biocida tipo 3** (higiene veterinaria con función biocida), es obligatoria la notificación a Sanidad de conformidad con la **Disposición Transitoria Segunda del RD 1054/2002** (no requiere ensayos de eficacia).  
  Más información: [Notificación DT2 - Ministerio de Sanidad](https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regPlaguicidas/dt2notificanuevo.htm)\n

En todo caso, para los casos anteriores, una vez que las sustancias activas que formen parte del producto (sustancias biocidas) cuenten con Reglamento de Ejecución para los tipos de productos biocidas que se quieren defender, esos productos deberán solicitar su registro por procedimiento europeo, de conformidad con las exigencias del Reglamento (UE) 528/2012.

En todo caso, si los productos que se deseen comercializar estén afectados o no por lo indicado anteriormente, son productos químicos peligrosos (mezclas o sustancias) quedarán afectados por la normativa de clasificación y etiquetado de mezclas y sustancias químicas, debiendo estar debidamente etiquetados, contar con ficha de datos de seguridad (FDS) y ser notificados a toxicología a través de un expediente PCN.

Por tanto, tal y como recomiendan desde ASEMAZ, lo más conveniente es poneros en contacto con la autoridad competente correspondiente para que os puedan dar información detallada."""
    ],
    "vitamina a": [
        """De acuerdo con el Reglamento 1223/2009, para cualquier producto cosmético que contenga las sustancias 'Retinol', 'Retinyl Acetate' o 'Retinyl Palmitate', la mención **“Este producto contiene vitamina A. Tenga en cuenta su ingesta diaria antes de utilizarlo”** es obligatoria. 
Por tanto, la advertencia debe figurar literalmente en el etiquetado del producto.""",
        
        """Entendemos que esta advertencia pueda generar cierta confusión en el consumidor, pero modificar la redacción obligatoria no es una opción, ya que debe figurar exactamente con la redacción establecida en el Reglamento. 
No obstante, y siempre bajo criterio del evaluador de seguridad del producto, puede añadirse una advertencia complementaria que aclare que el producto es de uso cosmético y no debe ingerirse."""
    ]
}

# ==============================================
# RESPUESTAS DE REDIRECCIÓN PREDEFINIDAS
# ==============================================
REDIRECCIONES_PREDEFINIDAS = {
    "internacional": {
        "palabras": [
            "exportar", "exportación", "terceros países", "fuera de la ue",
            "australia", "nueva zelanda", "ee.uu", "eeuu", "china", "reino unido"
        ],
        "respuesta": """\
Buenos días,

Para consultas relacionadas con terceros países pueden ayudaros mis compañeras del área internacional.  
Lamentablemente, ellas aún no tienen acceso a la plataforma de Consultas Técnicas,  
pero puedes escribirles a la siguiente dirección de correo electrónico:

[stanpainternacional@stanpa.com](mailto:stanpainternacional@stanpa.com)

Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.

Reciba un cordial saludo,  
Departamento Técnico.
"""
    },
    "sostenibilidad": {
        "palabras": [
            "sostenibilidad", "medio ambiente", "huella", "ecodiseño",
            "envase sostenible", "packaging sostenible"
        ],
        "respuesta": """\
Buenos días,

En relación con tu consulta, lamentamos informarte que la responsable de Sostenibilidad,  
quien podría ayudarte, no tiene acceso a la nueva plataforma de consultas técnicas.  
No obstante, puedes dirigirte a ella a través del siguiente correo electrónico:

[lucia.jimenez@stanpa.com](mailto:lucia.jimenez@stanpa.com)

Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.

Reciba un cordial saludo,  
Departamento Técnico.
"""
    }
}

# ==============================================
# 5️⃣ FUNCIÓN PRINCIPAL
# ==============================================
def responder_chatbot(pregunta, mostrar_contexto=False):
    from datetime import datetime
    hora = datetime.now().hour
    saludo = "Buenos días," if hora < 12 else "Buenas tardes,"
    despedida = (
        "<br><br>Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.<br><br>"
        "Reciba un cordial saludo,<br>"
        "Departamento Técnico."
    )

    pregunta_sin_acentos = quitar_acentos(pregunta.lower())

    # 🔹 1️⃣ Redirecciones predefinidas (internacional, sostenibilidad, etc.)
    for area, datos in REDIRECCIONES_PREDEFINIDAS.items():
        for palabra in datos["palabras"]:
            # Coincidencia robusta (palabra exacta, aunque esté seguida de coma o punto)
            if re.search(rf"\b{re.escape(palabra)}\b", pregunta_sin_acentos):
                return datos["respuesta"]

    # 🔹 2️⃣ Temas fijos (vitamina A, cosmética animal, etc.)
    if any(p in pregunta_sin_acentos for p in ["vitamina a", "retinol", "retinil"]):
        texto = "\n\n".join(FRASES_POR_TEMA["vitamina a"])
        return f"{saludo}\n\n{texto}\n\n{despedida}"

    if any(p in pregunta_sin_acentos for p in [
        "cosmetica animal", "cosmetica para animales", "higiene animal",
        "cuidado animal", "cosmetica veterinaria", "productos para mascotas"
    ]):
        texto = FRASES_POR_TEMA["cosmetica para animales"][0]
        return f"{saludo}\n\n{texto}\n\n{despedida}"
    
    # 🔹 3️⃣ Caso general: embeddings + GPT
    fragmentos = buscar_contexto(pregunta)
    contexto = "\n\n".join(fragmentos) if fragmentos else ""
    prompt = f"""
Eres un asistente técnico experto en legislación cosmética, biocidas y productos regulados.
Redacta una respuesta formal, precisa y técnica, pero **no incluyas fórmulas de cortesía como 'Estimado/a' ni nombres del remitente.**
Tampoco incluyas una firma con nombres personales; la respuesta debe cerrarse con 'Departamento Técnico.'
Empieza la respuesta directamente tras el saludo y no incluyas saludos ni cierres redundantes.

Contexto normativo: {contexto}
Pregunta: {pregunta}
"""

    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    ).choices[0].message.content.strip()

    # --- Ajuste final ---
    if not respuesta.lower().startswith(("buenos días", "buenas tardes")):
        respuesta = f"{saludo}\n\n{respuesta}"

    # 🧹 Eliminar cualquier cierre redundante del modelo
    for texto_final in ["departamento técnico", "reciba un cordial saludo"]:
        if texto_final in respuesta.lower():
            respuesta = respuesta[:respuesta.lower().rfind(texto_final)].strip()
            break  # detiene la limpieza en la primera coincidencia

    # 💬 Añadir siempre la despedida fija
    respuesta = f"{respuesta}\n\n{despedida}"

    return respuesta


# ==============================================
# 🖥️ INTERFAZ STREAMLIT
# ==============================================
st.set_page_config(page_title="Chatbot Regulatorio Interno", page_icon="💬", layout="centered")

st.markdown("""
    <style>
    .chat-response {
        background-color: #ffffff;
        border: 1px solid #e0e4e8;
        border-radius: 0.8rem;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 1rem;
        line-height: 1.7;
        font-size: 16px;
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>💬 Chatbot Regulatorio Interno</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Consultas sobre normativa cosmética y regulación</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if "historial" not in st.session_state:
    st.session_state.historial = []

for entrada in st.session_state.historial:
    if entrada["role"] == "user":
        st.markdown(f"<div class='chat-question'>🧴 <strong>Tú:</strong> {entrada['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(render_html_markdown(entrada["content"]), unsafe_allow_html=True)

pregunta = st.chat_input("Escribe tu consulta y pulsa Enter para enviar...")

if pregunta:
    st.session_state.historial.append({"role": "user", "content": pregunta})
    with st.spinner("Analizando consulta..."):
        respuesta = responder_chatbot(pregunta)
    st.session_state.historial.append({"role": "assistant", "content": respuesta})
    st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🧠 Basado en el histórico de consultas internas y el modelo GPT-4o de OpenAI.")