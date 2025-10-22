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

            # Si no encuentra nada específico, devuelve el fragmento original
            return frag.strip()

    return None

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
    st.error("❌ No se encontró el archivo 'emb_consultas_comprimido.npz'. Genera primero los embeddings con 'generar_embeddings_excel.py'.")
    st.stop()

# ==============================================
# 4️⃣ Buscar contexto relevante con embeddings
# ==============================================
def buscar_contexto(pregunta, top_k=5, umbral_similitud=0.78):
    """
    Busca los fragmentos de respuesta más relevantes según similitud semántica.
    Aplica un umbral de corte para evitar recuperar texto solo vagamente relacionado.
    """
    pregunta_sin_acentos = quitar_acentos(pregunta.lower())

    emb_pregunta = client.embeddings.create(
        model="text-embedding-3-small",
        input=pregunta_sin_acentos
    ).data[0].embedding

    similitudes = cosine_similarity([emb_pregunta], emb_consultas)[0]

    # Filtrar por umbral mínimo de similitud
    indices_validos = [i for i, s in enumerate(similitudes) if s >= umbral_similitud]
    if not indices_validos:
        # Si no hay coincidencias suficientemente fuertes, devolvemos las mejores dos
        indices_validos = similitudes.argsort()[-2:][::-1]

    # Ordenar los seleccionados por similitud descendente
    indices_ordenados = sorted(indices_validos, key=lambda i: similitudes[i], reverse=True)[:top_k]
    fragmentos = [pares[i][1] for i in indices_ordenados]

    # Refinar: evitar fragmentos que contengan "uñas", "depilación", "perfume", etc., si la pregunta no lo menciona
    temas_no_relevantes = ["uñas", "depilación", "perfume", "peluquería", "barniz"]
    if not any(t in pregunta_sin_acentos for t in temas_no_relevantes):
        fragmentos = [f for f in fragmentos if not any(t in f.lower() for t in temas_no_relevantes)]

    return fragmentos   

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

        **Registro nacional:**
        - Si se trata de un zoosanitario para uso en entorno ganadero (insecticida, larvicida, desinfectante, etc.), deberá solicitarse su registro ante el MAPA como plaguicida, con los correspondientes ensayos según la eficacia que se quiera defender. Más información: https://www.mapa.gob.es/es/ganaderia/temas/sanidad-animal-higiene-ganadera/Higiene-de-la-produccion-primaria-ganadera/registro-de-productos-zoosanitarios/
        - Si se trata de un plaguicida no agrícola (desinfectante de uso en la industria alimentaria o uso ambiental, rodenticida, etc.), deberá solicitarse su registro ante Sanidad como plaguicida no agrícola. Más información: https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regNacional/requisitos_nacional.htm
        - Si se trata de un biocida tipo 3 (higiene veterinaria con función biocida), es obligatoria la notificación a Sanidad de conformidad con la Disposición Transitoria Segunda del RD 1054/2002 (no requiere ensayos de eficacia). Más información: https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regPlaguicidas/dt2notificanuevo.htm

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
# 🔹 RESPUESTAS DE REDIRECCIÓN PREDEFINIDAS
# ==============================================
REDIRECCIONES_PREDEFINIDAS = {
    "internacional": {
        "palabras": [
            "exportar", "exportación", "terceros países", "fuera de la ue",
            "australia", "nueva zelanda", "ee.uu", "eeuu", "china", "reino unido"
        ],
        "respuesta": """
Buenos días,<br><br>
Para consultas relacionadas con terceros países pueden ayudaros mis compañeras del área internacional. 
Lamentablemente, ellas aún no tienen acceso a la plataforma de Consultas Técnicas, 
pero puedes escribirles a la dirección de correo electrónico:<br>
<a href="mailto:stanpainternacional@stanpa.com" style="color:#0078D7; font-weight:bold; text-decoration:none;">
stanpainternacional@stanpa.com</a><br><br>
Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.<br>
Reciba un cordial saludo,<br>
Departamento Técnico.
"""
    },
    "sostenibilidad": {
        "palabras": [
            "sostenibilidad", "medio ambiente", "huella", "ecodiseño", 
            "envase sostenible", "packaging sostenible"
        ],
        "respuesta": """
Buenos días,<br><br>
En relación con tu consulta, lamentamos informarte que la responsable de Sostenibilidad, 
quien podría ayudarte, no tiene acceso a la nueva plataforma de consultas técnicas. 
No obstante, puedes dirigirte a ella a través del siguiente correo electrónico:<br>
<a href="mailto:lucia.jimenez@stanpa.com" style="color:#0078D7; font-weight:bold; text-decoration:none;">
lucia.jimenez@stanpa.com</a><br><br>
Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.<br>
Reciba un cordial saludo,<br>
Departamento Técnico.
"""
    }
}

# --- 5️⃣ Generar respuesta con GPT ---
def responder_chatbot(pregunta, mostrar_contexto=False):
    from datetime import datetime

    # --- 🕒 Determinar saludo según hora ---
    hora = datetime.now().hour
    saludo = "Buenos días," if hora < 12 else "Buenas tardes,"

    # --- 🧾 Despedida fija ---
    despedida = (
        "Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.<br><br>"
        "Reciba un cordial saludo,<br>"
        "Departamento Técnico."
    )

    pregunta_lower = pregunta.lower()
    pregunta_sin_acentos = quitar_acentos(pregunta_lower)

    # --- 🧭 Detección automática de redirecciones fijas ---
    for area, datos in REDIRECCIONES_PREDEFINIDAS.items():
        if any(p in pregunta_sin_acentos for p in datos["palabras"]):
            return datos["respuesta"]

    # --- 🔍 Detección prioritaria de temas normativos fijos ---
    if "cosmetica para animales" in pregunta_sin_acentos or "cosmetica animal" in pregunta_sin_acentos:
        fragmentos = []  # forzar respuesta por tema
    else:
        fragmentos = buscar_contexto(pregunta)

    # ⚙️ Si no hay contexto relevante, continuar igualmente con detecciones temáticas
    if not fragmentos:
        fragmentos = []

    contexto = "\n\n".join(fragmentos)

    # --- 🧩 Detección de redirección ---
    if detectar_redireccion(fragmentos):
        redireccion_texto = extraer_redireccion(fragmentos)
        if redireccion_texto:
            return f"{saludo}\n\n{redireccion_texto}\n\n{despedida}"
        else:
            return (
                f"{saludo}\n\nLa consulta que plantea corresponde a otro ámbito de competencia. "
                "Le recomendamos trasladarla al departamento o autoridad competente indicada en los procedimientos internos.\n\n"
                f"{despedida}"
            )

    # --- Detectar tema ---
    frases_relevantes = []
    for tema, frases in FRASES_POR_TEMA.items():
        if tema in pregunta_sin_acentos:
            frases_relevantes.extend(frases)

    # --- 🧠 Detección avanzada para cosmética animal ---
    palabras_clave_animales = [
        "cosmetica animal", "cosmetica para animales", "cosmeticos animales",
        "productos cosmeticos destinados a animales", "productos destinados a animales",
        "fabricar cosmeticos para animales", "fabricar productos cosmeticos destinados a animales",
        "fabricacion cosmetica para animales", "cosmetica veterinaria",
        "higiene animal", "cuidado animal", "cosmetica para mascotas", "productos para mascotas"
    ]

    es_cosmetica_animal = any(p in pregunta_sin_acentos for p in palabras_clave_animales)

    if any(p in pregunta_sin_acentos for p in ["animal", "animales"]) and any(k in pregunta_sin_acentos for k in ["fabricacion", "fabricar", "declaracion responsable", "registro"]):
        es_cosmetica_animal = True

    if not es_cosmetica_animal:
        try:
            emb_pregunta = client.embeddings.create(
                model="text-embedding-3-small",
                input=pregunta
            ).data[0].embedding

            emb_animal = client.embeddings.create(
                model="text-embedding-3-small",
                input="cosmética para animales, productos cosméticos destinados a animales, higiene animal, fabricación de cosméticos para animales, cosmética veterinaria"
            ).data[0].embedding

            similitud = cosine_similarity([emb_pregunta], [emb_animal])[0][0]
            if similitud > 0.75:
                es_cosmetica_animal = True
        except Exception as e:
            print("⚠️ Error en detección semántica de cosmética animal:", e)

    if es_cosmetica_animal:
        # Obtener el texto original del tema desde FRASES_POR_TEMA
        texto_animal = FRASES_POR_TEMA.get("cosmetica para animales", ["Texto no disponible."])[0]

        # Convertirlo a formato HTML (cambia saltos de línea y negritas)
        texto_animal_html = (
            texto_animal
            .replace("\n", "<br>")
            .replace("**", "<b>").replace("<b><b>", "<b>").replace("</b></b>", "</b>")
        )

        respuesta_directa = f"""
        {saludo}<br><br>
        {texto_animal_html}<br><br>
        {despedida}
        """
        return respuesta_directa



    # --- 🧩 Filtrar la definición general cuando no aporta valor ---
    palabras_clave_ingredientes = [
        "formaldehido", "fenoxietanol", "metanol", "retinol", "plomo", "parabenos",
        "filtros uv", "filtro uv", "perfume", "fragancia", "conservante", "colorante",
        "nanomaterial", "biocida", "ingrediente", "sustancia", "compuesto", "aditivo", "alergeno"
    ]

    es_pregunta_de_ingrediente = any(p in pregunta_sin_acentos for p in palabras_clave_ingredientes)

    if not es_pregunta_de_ingrediente:
        try:
            emb_pregunta = client.embeddings.create(
                model="text-embedding-3-small",
                input=pregunta
            ).data[0].embedding

            emb_ingrediente = client.embeddings.create(
                model="text-embedding-3-small",
                input="preguntas sobre ingredientes cosméticos, sustancias prohibidas, conservantes, colorantes o compuestos químicos usados en cosmética"
            ).data[0].embedding

            similitud_ing = cosine_similarity([emb_pregunta], [emb_ingrediente])[0][0]
            if similitud_ing > 0.75:
                es_pregunta_de_ingrediente = True
        except Exception as e:
            print("⚠️ Error en detección semántica de ingredientes:", e)

    if es_pregunta_de_ingrediente and "cosmetico" in FRASES_POR_TEMA:
        frases_relevantes = [f for f in frases_relevantes if f not in FRASES_POR_TEMA["cosmetico"]]

    # --- Generar prompt final ---
    frases_texto = "\n".join([f"- {f}" for f in frases_relevantes]) if frases_relevantes else ""
    if es_cosmetica_animal:
        frases_texto = "\n".join(["- " + " ".join(FRASES_POR_TEMA.get("cosmetica para animales", []))])

    prompt = f"""
    Eres un asistente experto en legislación cosmética, biocidas y productos regulados.
    ...
    """
    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    ).choices[0].message.content.strip()

    # --- Ajustes finales ---
    respuesta_limpia = respuesta.strip()
    if not respuesta_limpia.lower().startswith(("buenos días", "buenas tardes", "buenas noches")):
        respuesta_limpia = f"{saludo}\n\n{respuesta_limpia}"
    if "departamento técnico" not in respuesta_limpia.lower():
        respuesta_limpia = f"{respuesta_limpia}\n\n{despedida}"

    return respuesta_limpia

# ==============================================
# 🖥️ INTERFAZ STREAMLIT (versión moderna tipo chat)
# ==============================================
st.set_page_config(
    page_title="Chatbot Regulatorio Interno",
    page_icon="💬",
    layout="centered"
)

# --- Estilos modernos ---
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

# --- Cabecera ---
st.markdown("<h1 style='text-align:center;'>💬 Chatbot Regulatorio Interno</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Consultas sobre normativa cosmética, biocidas y productos regulados</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Inicializar historial de conversación ---
if "historial" not in st.session_state:
    st.session_state.historial = []

# --- Mostrar conversación previa ---
# --- Mostrar conversación previa ---
for entrada in st.session_state.historial:
    if entrada["role"] == "user":
        st.markdown(f"<div class='chat-question'>🧴 <strong>Tú:</strong> {entrada['content']}</div>", unsafe_allow_html=True)
    else:
        # ✅ Renderiza correctamente HTML y Markdown dentro del contenedor blanco
        st.markdown(f"<div class='chat-response'>{entrada['content']}</div>", unsafe_allow_html=True)

# --- Entrada tipo chat (Enter → enviar, Shift+Enter → salto de línea) ---
pregunta = st.chat_input("Escribe tu consulta y pulsa Enter para enviar...")

if pregunta:
    # Mostrar pregunta del usuario
    st.session_state.historial.append({"role": "user", "content": pregunta})
    with st.spinner("Analizando consulta..."):
        respuesta = responder_chatbot(pregunta)
    st.session_state.historial.append({"role": "assistant", "content": respuesta})
    st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🧠 Basado en el histórico de consultas internas y el modelo GPT-4o de OpenAI.")