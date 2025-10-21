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
def buscar_contexto(pregunta, top_k=5):
    """Busca las respuestas más similares semánticamente."""
    pregunta_sin_acentos = quitar_acentos(pregunta.lower())
    emb_pregunta = client.embeddings.create(
        model="text-embedding-3-small",
        input=pregunta_sin_acentos
    ).data[0].embedding
    similitudes = cosine_similarity([emb_pregunta], emb_consultas)[0]
    indices = similitudes.argsort()[-top_k:][::-1]
    fragmentos = [pares[i][1] for i in indices]
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
    ]
}

# ==============================================
# 🔹 RESPUESTAS DE REDIRECCIÓN PREDEFINIDAS
# ==============================================
REDIRECCIONES_PREDEFINIDAS = {
    "internacional": {
        "palabras": ["exportar", "exportación", "terceros países", "fuera de la ue",
                     "australia", "nueva zelanda", "ee.uu", "eeuu", "china", "reino unido"],
        "respuesta": """Buenos días,

    Para consultas relacionadas con terceros países pueden ayudaros mis compañeras del área internacional. Lamentablemente, ellas aún no tienen acceso a la plataforma de Consultas Técnicas, pero puedes escribirle a la dirección de correo electrónico:
    **stanpainternacional@stanpa.com**

    Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.  
    Recibe un cordial saludo,  
    Departamento Técnico."""
    },
    "sostenibilidad": {
        "palabras": ["sostenibilidad", "medio ambiente", "huella", "ecodiseño", "envase sostenible", "packaging sostenible"],
        "respuesta": """Buenos días,

        En relación con tu consulta, lamentamos informarte que la responsable de Sostenibilidad, quien podría ayudarte, no tiene acceso a la nueva plataforma de consultas técnicas. No obstante, puedes dirigirte a ella a través del siguiente correo electrónico:
        **lucia.jimenez@stanpa.com**

        Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.  
        Recibe un cordial saludo,  
        Departamento Técnico."""
    }
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
    pregunta_sin_acentos = quitar_acentos(pregunta_lower)

    # --- 🧭 Detección automática de redirecciones fijas ---
    for area, datos in REDIRECCIONES_PREDEFINIDAS.items():
        if any(p in pregunta_sin_acentos for p in datos["palabras"]):
            return datos["respuesta"]

    fragmentos = buscar_contexto(pregunta)
    if not fragmentos:
        return "No encontré información relevante en la base de datos. ¿Podrías reformular la pregunta?"

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

    contexto = "\n\n".join(fragmentos)

    # --- Detectar tema ---
    frases_relevantes = []
    for tema, frases in FRASES_POR_TEMA.items():
        if tema in pregunta_sin_acentos:
            frases_relevantes.extend(frases)
    
    # --- 🧠 Detección avanzada para cosmética animal ---
    palabras_clave_animales = [
        "cosmetica animal", "cosmetica para animales", "cosmeticos animales", "cosmeticos destinados a animales",
        "productos cosmeticos destinados a animales", "productos destinados a animales",
        "fabricar cosmeticos para animales", "fabricar productos cosmeticos destinados a animales",
        "fabricación cosmetica para animales", "cosmetica veterinaria",
        "higiene animal", "cuidado animal", "cosmetica para mascotas", "productos para mascotas"
    ]

    es_cosmetica_animal = any(p in pregunta_sin_acentos for p in palabras_clave_animales)

    # Si hay mención de animales + fabricación o declaración -> forzar cosmética animal
    if any(p in pregunta_sin_acentos for p in ["animal", "animales"]) and any(k in pregunta_sin_acentos for k in ["fabricación", "fabricar", "declaración responsable", "registro"]):
        es_cosmetica_animal = True

    # Si no coincide por palabra, comprobar similitud semántica con embeddings
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
        frases_relevantes.extend(FRASES_POR_TEMA.get("cosmetica para animales", []))
    
    # --- 🧩 Filtrar la definición general cuando no aporta valor ---

    # Palabras clave comunes relacionadas con ingredientes o sustancias
    palabras_clave_ingredientes = [
        "formaldehido", "fenoxietanol", "metanol", "retinol", "plomo", "parabenos", "filtros uv", "filtro uv", "perfume", "fragancia",
        "conservante", "colorante", "nanomaterial", "biocida", "ingrediente", "sustancia", "compuesto", "aditivo", "alergeno"
    ]

    # Detección inicial por palabra
    es_pregunta_de_ingrediente = any(p in pregunta_sin_acentos for p in palabras_clave_ingredientes)


    # Si no hay coincidencias claras, analizar similitud semántica
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

    # Si la pregunta trata de ingredientes, eliminar la definición general del cosmético
    if es_pregunta_de_ingrediente and "cosmetico" in FRASES_POR_TEMA:
        frases_relevantes = [
            f for f in frases_relevantes
            if f not in FRASES_POR_TEMA["cosmetico"]
        ]

    # Si el tema detectado es cosmética animal, unir todas las frases en una sola para asegurar que el bloque completo se incluye
    frases_texto = "\n".join([f"- {f}" for f in frases_relevantes]) if frases_relevantes else ""

    if es_cosmetica_animal:
        frases_texto = "\n".join(["- " + " ".join(FRASES_POR_TEMA.get("cosmetica para animales", []))])

    prompt = f"""
    Eres un asistente experto en legislación cosmética, biocidas y productos regulados.

    Debes redactar una respuesta **formal, precisa y actualizada**, en tono técnico.
    Estructura la respuesta de la siguiente forma:

    1️⃣ Comienza con una **afirmación clara y objetiva** sobre la situación normativa del tema preguntado.
    2️⃣ Desarrolla a continuación una explicación completa con el contexto legal y técnico.
    3️⃣ Finaliza con la despedida establecida.

    La respuesta debe empezar con un saludo (“Buenos días,” / “Buenas tardes,”) y finalizar con:

    "Espero haber sido de utilidad y si necesita alguna cosa más, estamos a su disposición.
    Reciba un cordial saludo,
    Departamento Técnico."

    ⚖️ Instrucciones:
    - No inventes ni reformules información.
    - No incluyas recomendaciones ni valoraciones personales.
    - **Debes incluir en la respuesta todas las frases normativas listadas a continuación, sin omitir ninguna parte, sin resumir ni reescribirlas.**
    - Cada una de ellas debe aparecer *exactamente como está redactada* (sin comillas), en cursiva, dentro del texto final:
    {frases_texto}
    - **Mantén la estructura y el formato técnico** de la información (por ejemplo, listas con guiones, subtítulos en negrita como “Registro nacional”, saltos de línea, etc.).
    - Si la información incluye secciones con guiones o subtítulos, reprodúcelas con formato Markdown igual al original.
    - No transformes las listas en párrafos corridos.

    - Inserta las frases donde encajen naturalmente en el desarrollo.
    - El resto del texto debe complementar las frases con explicaciones objetivas y actuales.

    ---
    Contexto normativo (solo para ampliar datos coherentes con las frases anteriores):
    {contexto}

    ---
    Pregunta:
    {pregunta}
    """
    prompt += "\n\nRecuerda: conserva la estructura original (listas, títulos y saltos de línea) del texto normativo."

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
            respuesta = respuesta.replace("\n", "<br>")  # 👈 convierte saltos en HTML
        st.markdown(f"<div style='font-size:16px; line-height:1.6;'>{respuesta}</div>", unsafe_allow_html=True)
    else:
        st.warning("Por favor, escribe una consulta antes de enviar.")

st.markdown("---")
st.caption("🧠 Basado en el histórico de consultas internas y el modelo GPT-4o de OpenAI.")
