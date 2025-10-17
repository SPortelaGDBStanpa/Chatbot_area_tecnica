# ================================================================
# 💬 Chatbot Regulatorio Interno (Versión Streamlit con Agentes)
# ================================================================

import os
import streamlit as st
import openai

# --- Configuración inicial ---
openai.api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Chatbot Regulatorio Interno", page_icon="💬")

# ================================================================
# 1️⃣ AGENTES ESPECIALIZADOS
# ================================================================

class AgenteHigieneAnimal:
    """Asistente para productos de higiene/cuidado animal."""
    def __init__(self):
        self.frases_vigentes = [
    
        ]

# ================================================================
# 💬 Chatbot Regulatorio Interno (versión ampliada con agente específico animal)
# ================================================================

import os
import streamlit as st
import openai

# --- Configuración inicial ---
openai.api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="Chatbot Regulatorio Interno", page_icon="💬")

# ================================================================
# 1️⃣ AGENTES ESPECIALIZADOS
# ================================================================

class AgenteHigieneAnimal:
    """Asistente específico para productos cosméticos o de higiene destinados a animales."""
    def __init__(self):
        self.frases_vigentes = [
            """Un producto cosmético según el Reglamento (CE) 1223/2009 es toda sustancia o mezcla destinada a ser puesta en contacto con las partes superficiales del cuerpo humano (epidermis, sistema piloso y capilar, uñas, labios, órganos genitales externos) o con los dientes y mucosas bucales, con el fin exclusivo o principal de limpiarlos, perfumarlos, modificar su aspecto, protegerlos, mantenerlos en buen estado o corregir los olores corporales.
Por tanto, los productos destinados a la higiene animal no se consideran cosméticos según el Reglamento 1223/2009 y quedan fuera de su ámbito de aplicación.
Te pongo en contexto la situación de estos productos:
En un principio los productos cosméticos destinados a animales estaban considerados productos zoosanitarios. Con la publicación del Real Decreto 867/2020 estos productos quedaron fuera de su ámbito de aplicación. Sin embargo, en 2023 se publicó una sentencia del Tribunal Supremo que anulaba el artículo 1 (párrafos primero y segundo, incluyendo la primera frase del segundo párrafo), así como la Disposición Adicional Primera del citado Real Decreto.
A raíz del recurso interpuesto por la Asociación Nacional para la Salud Animal (ASEMAZ-ASA), los productos cosméticos destinados a animales volvieron a ser considerados productos zoosanitarios, por lo que requerían autorización y registro por parte del Ministerio de Agricultura y Pesca.
Ahora bien, con la publicación de la Ley 1/2025, de 1 de abril, de prevención de las pérdidas y el desperdicio alimentario, que modifica la Ley 8/2003, de sanidad animal, se elimina definitivamente la obligatoriedad de registro de los productos de higiene, cuidado y manejo de los animales (HCM), así como del material y utillaje zoosanitario (MUZ) para su comercialización. Por tanto, estos productos quedarían fuera del ámbito de competencias del Ministerio.
Ante esta situación, el pasado mes de junio nos pusimos en contacto con ASEMAZ, quienes nos informaron de lo siguiente:
Efectivamente, con la publicación de la Ley 1/2015, determinados productos zoosanitarios destinados a higiene, cuidado y manejo de los animales ya no tienen que ser notificados por el titular de los mismos para su comercialización.
Ahora bien, decimos “determinados” dado que dependiendo del “claim” reivindicado por el producto (biocidas), tendrán las siguientes obligaciones:
Registro nacional:
Si se trata de un zoosanitario para uso en entorno ganadero (insecticida, larvicida, desinfectante, etc..) tendrá que solicitarse su registro ante el MAPA como plaguicida, con sus correspondientes ensayos según la eficacia que se quiera defender. La página del MAPA donde podéis informaros es:
https://www.mapa.gob.es/es/ganaderia/temas/sanidad-animal-higiene-ganadera/Higiene-de-la-produccion…
Si se trata de un plaguicida no agrícola (desinfectante de uso en la industria alimentaria o uso ambiental, rodenticida, etc..) tendrá que solicitarse su registro ante el SANIDAD como plaguicida no agrícola, con sus correspondientes ensayos según la eficacia que se quiera defender. La página de SANIDAD donde podéis informaros es:
https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regNacional/requisitos_nacional…
Si se trata de un biocida tipo 3 (higiene veterinaria, pero con función biocida), es obligatorio hacer una notificación a Sanidad de conformidad con la Disposición Transitoria Segunda del RD 1054/2002 (no requiere ensayos de eficacia). El enlace de Sanidad donde encontráis la información es el siguiente:
https://www.sanidad.gob.es/areas/sanidadAmbiental/biocidas/registro/regPlaguicidas/dt2notificanuevo…
En todo caso, para los casos anteriores, una vez que las sustancias activas que formen parte del producto (sustancias biocidas) cuenten con Reglamento de Ejecución para los tipos de productos biocidas que se quieren defender, esos productos deberán solicitar su registro por procedimiento europeo, de conformidad con las exigencias del Reglamento (UE) 528/2012.
En todo caso, si los productos que se deseen comercializar estén afectados o no por lo indicado anteriormente, son productos químicos peligrosos (mezclas o sustancias) quedarán afectados por la normativa de clasificación y etiquetado de mezclas y sustancias químicas, debiendo estar debidamente etiquetados, contar con ficha de datos de seguridad (FDS) y ser notificados a toxicología a través de un expediente PCN.
No obstante, consideramos que lo conveniente es que vuestra empresa realice la consulta a la autoridad competente correspondiente, para que proporcionen una opinión fundada sobre los productos que desean fabricar/comercializar."""
        ]

    def responder(self, pregunta):
        frases_relevantes = [
            f"*{f}*" for f in self.frases_vigentes
            if any(pal in pregunta.lower() for pal in ["animal", "fabricar", "fabricación", "cosmética", "zoosanitario", "veterinario"])
        ]

        # Si no hay coincidencias, muestra todas
        if not frases_relevantes:
            frases_relevantes = [f"*{f}*" for f in self.frases_vigentes]

        texto = " ".join(frases_relevantes)
        return f"Según la normativa vigente: {texto}\n\nReciba un cordial saludo,\nDepartamento Técnico."


# ================================================================
# 2️⃣ CHATBOT PRINCIPAL
# ================================================================

class ChatbotRegulatorio:
    def __init__(self):
        self.agente_higiene = AgenteHigieneAnimal()

    def seleccionar_agente(self, pregunta):
        texto = pregunta.lower()

        if any(pal in texto for pal in ["animal", "zoosanitario", "veterinario", "mascotas", "perros", "gatos"]):
            return self.agente_higiene
        elif any(pal in texto for pal in ["biocida", "tp3", "tp4", "plaguicida", "desinfectante"]):
            return self.agente_biocidas
        elif any(pal in texto for pal in ["cosmético", "aemps", "cpnp", "1223/2009", "piel", "producto"]):
            return self.agente_cosmetica
        else:
            return self.agente_general

    def responder(self, pregunta):
        agente = self.seleccionar_agente(pregunta)
        return agente.responder(pregunta)


# ================================================================
# 3️⃣ INTERFAZ STREAMLIT
# ================================================================

st.title("💬 Chatbot Regulatorio Interno")
st.markdown("Consulta dudas técnicas sobre normativa cosmética, biocidas o higiene animal.")

bot = ChatbotRegulatorio()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pregunta := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    respuesta = bot.responder(pregunta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    with st.chat_message("assistant"):
        st.markdown(respuesta)

