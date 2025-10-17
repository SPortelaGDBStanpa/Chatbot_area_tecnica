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

En un principio los productos cosméticos destinados a animales estaban considerados productos zoosanitarios. Con la publicación del Real Decreto 867/2020 estos productos quedaron fuera de su ámbito de aplicación. Sin embargo, en 2023 se publicó una sentencia del Tribunal Supremo que anulaba parcialmente dicho real decreto.

Con la publicación de la Ley 1/2025, de 1 de abril, se elimina definitivamente la obligatoriedad de registro de los productos de higiene, cuidado y manejo de los animales (HCM), así como del material y utillaje zoosanitario (MUZ). Por tanto, estos productos quedan fuera del ámbito de competencias del Ministerio."""
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
