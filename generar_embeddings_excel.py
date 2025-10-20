# ==============================================
# 🧠 Generador de Embeddings para el Chatbot Regulatorio
# ==============================================

import os
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm  # barra de progreso visual

# --- 1️⃣ Configurar la API Key ---
# Asegúrate de tener la variable OPENAI_API_KEY en tu entorno
# En Windows (PowerShell): setx OPENAI_API_KEY "tu_clave_aqui"
# En macOS/Linux: export OPENAI_API_KEY="tu_clave_aqui"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 2️⃣ Cargar el Excel ---
ruta_excel = "C:/Users/susana.portela/OneDrive - Asociación Nacional de Perfumería y Cosmética/Documentos/Chatbot_area_tecnica/conversaciones_revisando.xlsx"

print(f"📂 Cargando archivo Excel: {ruta_excel}")
df = pd.read_excel(ruta_excel)
df.columns = df.columns.str.strip().str.lower()

# Tomar solo las consultas de usuario
consultas = df[df["role"].str.lower() == "user"]["content"].dropna().tolist()
print(f"🧾 Total de consultas detectadas: {len(consultas)}")

# --- 3️⃣ Generar los embeddings por lotes ---
def generar_embeddings_por_lotes(textos, batch_size=50):
    """Genera embeddings en lotes para evitar saturar la API."""
    embeddings = []
    for i in tqdm(range(0, len(textos), batch_size), desc="Generando embeddings"):
        batch = textos[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings.extend([d.embedding for d in response.data])
    return np.array(embeddings)

print("⚙️ Generando embeddings, puede tardar unos minutos...")
embeddings = generar_embeddings_por_lotes(consultas)

# --- 4️⃣ Guardar embeddings en archivo numpy ---
archivo_salida = "emb_consultas.npy"
np.save(archivo_salida, embeddings)
print(f"✅ Embeddings guardados en: {archivo_salida}")
print("🎯 Ahora puedes cargar este archivo en tu app con:")
print("    emb_consultas = np.load('emb_consultas.npy')")
