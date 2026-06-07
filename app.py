import streamlit as st
import json
from pathlib import Path
from datetime import datetime
from typing import List
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import os

class EmbeddingsLocales(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

# Configuración
CHROMA_DIR = Path("data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

PREGUNTAS_REPORTE = [
    {
        "id": "movimientos_estrategicos",
        "titulo": "Movimientos estratégicos recientes",
        "pregunta": "¿Qué movimientos estratégicos relevantes han hecho Falabella, Studio F y Zara Colombia en los últimos 90 días?",
    },
    {
        "id": "transformacion_digital",
        "titulo": "Transformación digital",
        "pregunta": "¿Cómo están respondiendo Falabella, Studio F y Zara a la transformación digital en Colombia?",
    },
    {
        "id": "expansion",
        "titulo": "Expansión y red de tiendas",
        "pregunta": "¿Cuál de las tres marcas está apostando más agresivamente por expansión en Colombia y por qué?",
    },
    {
        "id": "señales_producto",
        "titulo": "Señales de producto y colecciones",
        "pregunta": "¿Qué señales hay sobre nuevos productos, colecciones o cambios de portafolio en Falabella, Studio F y Zara Colombia?",
    },
    {
        "id": "dinamica_competitiva",
        "titulo": "Dinámica competitiva",
        "pregunta": "¿Qué tensiones o movimientos competitivos directos se observan entre Falabella, Studio F y Zara en Colombia?",
    },
    {
        "id": "señal_semana",
        "titulo": "Señal de la semana",
        "pregunta": "¿Cuál es el evento o movimiento más relevante ocurrido recientemente en el retail de moda colombiano y qué implica para el mercado?",
    }
]

PROMPT_SISTEMA = """Eres un analista de inteligencia de mercado especializado en retail de moda en Colombia.

Tu trabajo es analizar información de fuentes públicas sobre Falabella, Studio F y Zara Colombia, 
y producir inteligencia accionable para equipos de mercadeo y producto.

Fecha de hoy: {fecha_hoy}

CONTEXTO DE FUENTES:
{contexto}

INSTRUCCIONES:
- Basa tu análisis ÚNICAMENTE en la información del contexto proporcionado
- Si la información no está en el contexto, indícalo explícitamente
- Prioriza información reciente sobre información antigua
- Distingue entre hechos confirmados y señales o tendencias
- Cuando la pregunta involucre más de una marca, estructura el análisis 
  por marca primero y luego sintetiza las dinámicas comparativas
- Calibra tu confianza: si la evidencia es débil, dilo explícitamente
- Usa lenguaje de negocio, no técnico

FORMATO DE RESPUESTA:

Si la pregunta es sobre UNA marca:

**HALLAZGO PRINCIPAL**
[Una oración que resume lo más importante]

**EVIDENCIA**
[2-3 puntos concretos con fecha y fuente]

**IMPLICACIÓN PARA EL NEGOCIO**
[Qué debería considerar un equipo de mercadeo o producto]

**LIMITACIONES**
[Qué no sabes o qué información faltaría]

Si la pregunta es COMPARATIVA entre marcas:

**DINÁMICA DE MERCADO**
[Una oración que describe el patrón competitivo central]

**POR MARCA**
[Para cada marca relevante: hallazgo principal + 1-2 evidencias con fecha y fuente]

**MAPA COMPETITIVO**
[Tabla comparativa de las marcas en la dimensión analizada]

**IMPLICACIÓN ESTRATÉGICA**
[Qué significa esta dinámica para alguien que compite en este mercado]

**LIMITACIONES**
[Qué información faltaría para un análisis más robusto]
"""

def formatear_contexto(documentos) -> str:
    fragmentos = []
    for doc in documentos:
        fragmento = f"""
Fuente: {doc.metadata['fuente']}
Fecha: {doc.metadata['fecha']}
Marca: {doc.metadata['marca']}
---
{doc.page_content}
        """.strip()
        fragmentos.append(fragmento)
    return "\n\n".join(fragmentos)

@st.cache_resource
def cargar_recursos():
    embeddings = EmbeddingsLocales("all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    return vectorstore, llm

def construir_cadena(vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 9}
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_SISTEMA),
        ("human", "{pregunta}")
    ])
    cadena = (
        {
            "contexto": retriever | formatear_contexto,
            "pregunta": RunnablePassthrough(),
            "fecha_hoy": lambda _: datetime.now().strftime("%d de %B de %Y")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return cadena

# UI
st.set_page_config(
    page_title="Retail Intelligence Colombia",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Retail Intelligence Colombia")
st.caption(f"Inteligencia de mercado automatizada · Falabella · Studio F · Zara Colombia · {datetime.now().strftime('%d %b %Y')}")

st.divider()

vectorstore, llm = cargar_recursos()
cadena = construir_cadena(vectorstore, llm)

tab_reporte, tab_consulta = st.tabs(["📊 Reporte semanal", "💬 Consulta libre"])

with tab_reporte:
    st.subheader("Reporte de inteligencia semanal")
    st.caption("Análisis automatizado sobre las 6 dimensiones clave del mercado")

    if st.button("Generar reporte completo", type="primary"):
        for seccion in PREGUNTAS_REPORTE:
            with st.expander(f"**{seccion['titulo']}**", expanded=True):
                with st.spinner("Analizando fuentes..."):
                    respuesta = cadena.invoke(seccion["pregunta"])
                    st.markdown(respuesta)

with tab_consulta:
    st.subheader("Consulta libre")
    st.caption("Haz cualquier pregunta sobre el mercado de retail de moda en Colombia")

    pregunta = st.text_area(
        "Tu pregunta",
        placeholder="¿Qué estrategia de precios está siguiendo Falabella frente a Zara?",
        height=100
    )

    if st.button("Consultar", type="primary"):
        if pregunta.strip():
            with st.spinner("Analizando fuentes..."):
                respuesta = cadena.invoke(pregunta)
                st.markdown(respuesta)
        else:
            st.warning("Escribe una pregunta antes de consultar.")