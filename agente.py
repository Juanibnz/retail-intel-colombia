import json
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

# Configuración
CHROMA_DIR = Path("data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

def cargar_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    return vectorstore

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
- Calibra tu confianza: si la evidencia es débil o escasa, dilo explícitamente
  en lugar de generar recomendaciones específicas sin respaldo
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

---

Si la pregunta es COMPARATIVA entre marcas:

**DINÁMICA DE MERCADO**
[Una oración que describe el patrón competitivo central]

**POR MARCA**
[Para cada marca relevante: hallazgo principal + 1-2 evidencias con fecha y fuente]

**MAPA COMPETITIVO**
[Tabla o síntesis de cómo se posiciona cada marca en la dimensión analizada]

**IMPLICACIÓN ESTRATÉGICA**
[Qué significa esta dinámica para alguien que compite en este mercado]

**LIMITACIONES**
[Qué información faltaría para un análisis más robusto]
"""

def construir_cadena(vectorstore: Chroma):
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    
    # k=9 para comparativas: 3 fragmentos por marca
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

def consultar(cadena, pregunta: str) -> str:
    print(f"\n{'='*60}")
    print(f"PREGUNTA: {pregunta}")
    print('='*60)
    respuesta = cadena.invoke(pregunta)
    print(respuesta)
    return respuesta

def main():
    print("Cargando base de conocimiento...")
    vectorstore = cargar_vectorstore()
    
    print("Construyendo agente...")
    cadena = construir_cadena(vectorstore)
    
    # Preguntas de prueba
    preguntas = [
        "¿Qué movimientos estratégicos ha hecho Falabella en Colombia en el último año?",
        "¿Cómo está evolucionando la estrategia digital de Studio F?",
        "¿Qué señales hay sobre la expansión de Zara en Colombia?",
        "¿Cómo están respondiendo Falabella, Studio F y Zara a la transformación digital en Colombia?",
        "¿Cuál de las tres marcas está apostando más agresivamente por expansión en Colombia y por qué?"
    ]
    
    for pregunta in preguntas:
        consultar(cadena, pregunta)

if __name__ == "__main__":
    main()