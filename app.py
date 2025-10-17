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

class AgenteCosmeticaHumana:
    """Asistente para consultas sobre cosmética humana."""
    def __init__(self):
        self.frases_vigentes = [
            "Los productos cosméticos están regulados por el Reglamento (CE) 1223/2009, aplicable exclusivamente a productos destinados al uso humano.",
            "La fabricación requiere la presentación de una Declaración Responsable ante la AEMPS, conforme a las Buenas Prácticas de Fabricación (ISO 22716).",
            "Los productos deben notificarse en el Portal Europeo CPNP antes de su comercialización.",
            "El Responsable de Producto debe garantizar la seguridad del cosmético y disponer del expediente PIF."
        ]

    def responder(self, pregunta):
        frases = "\n".join([f"*{f}*" for f in self.frases_vigentes])
        return (
            f"Según la normativa vigente:\n{frases}\n\n"
            "Reciba un cordial saludo,\nDepartamento Técnico."
        )


class AgenteHigieneAnimal:
    """Asistente para productos de higiene/cuidado animal."""
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
        frases = "\n".join([f"*{f}*" for f in self.frases_vigentes])
        return (
            f"Según la información vigente:\n{frases}\n\n"
            "Reciba un cordial saludo,\nDepartamento Técnico."
        )


class AgenteBiocidas:
    """Asistente para productos biocidas (Reglamento 528/2012)."""
    def __init__(self):
        self.frases_vigentes = [
            "Los productos biocidas están regulados por el Reglamento (UE) 528/2012, que establece los requisitos para su autorización y comercialización en la Unión Europea.",
            "Cada producto debe contener únicamente sustancias activas aprobadas para el tipo de producto (TP) correspondiente.",
            "Mientras la sustancia activa esté en proceso de evaluación, puede aplicarse la Disposición Transitoria Segunda del RD 1054/2002, que permite su notificación simplificada."
        ]

    def responder(self, pregunta):
        frases = "\n".join([f"*{f}*" for f in self.frases_vigentes])
        return (
            f"De acuerdo con la normativa vigente:\n{frases}\n\n"
            "Reciba un cordial saludo,\nDepartamento Técnico."
        )


# ================================================================
# 2️⃣ CHATBOT PRINCIPAL
# ================================================================

class ChatbotRegulatorio:
    def __init__(self):
        self.agente_cosmetica = AgenteCosmeticaHumana()
        self.agente_higiene = AgenteHigieneAnimal()
        self.agente_biocidas = AgenteBiocidas()

    def seleccionar_agente(self, pregunta):
        """Detecta el tema y selecciona el agente adecuado."""
        texto = pregunta.lower()
        if any(pal in texto for pal in ["animal", "veterinario", "zoosanitario"]):
            return self.agente_higiene
        elif any(pal in texto for pal in ["biocida", "tp3", "tp4", "desinfectante", "plaguicida"]):
            return self.agente_biocidas
        elif any(pal in texto for pal in ["aemps", "cpnp", "humano", "piel", "cosmético", "1223/2009"]):
            return self.agente_cosmetica
        else:
            return None

    def responder(self, pregunta):
        """Redirige la consulta al agente adecuado o responde genéricamente."""
        agente = self.seleccionar_agente(pregunta)
        if agente:
            return agente.responder(pregunta)
        else:
            return (
                "No se ha identificado un agente especializado para esta consulta. "
                "Por favor, indique si se refiere a cosmética humana, biocidas o productos de higiene animal."
            )

# ================================================================
# 3️⃣ INTERFAZ STREAMLIT
# ================================================================

st.title("💬 Chatbot Regulatorio Interno")
st.markdown("Consulta dudas técnicas sobre normativa cosmética, biocidas o higiene animal.")

# Crear instancia del chatbot
bot = ChatbotRegulatorio()

# Historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
if pregunta := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Obtener respuesta del bot
    respuesta = bot.responder(pregunta)
    st.session_state.messages.append({"role": "assistant", "content": respuesta})

    with st.chat_message("assistant"):
        st.markdown(respuesta)
