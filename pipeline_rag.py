import json
import os
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class EmbeddingsLocales(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

# Configuración
DATA_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def cargar_articulos() -> list[Document]:
    """Carga todos los JSON y los convierte en Documents de LangChain."""
    documentos = []
    
    for archivo in DATA_DIR.glob("*.json"):
        with open(archivo, "r", encoding="utf-8") as f:
            articulos = json.load(f)
        
        for articulo in articulos:
            # Construir texto completo del documento
            texto = f"""
Marca: {articulo['marca']}
Título: {articulo['titulo']}
Fecha: {articulo['fecha']}
Fuente: {articulo['fuente']}

{articulo['resumen']}
            """.strip()
            
            # Metadata para filtrar después
            metadata = {
                "marca": articulo["marca"],
                "titulo": articulo["titulo"],
                "url": articulo["url"],
                "fecha": articulo["fecha"],
                "fuente": articulo["fuente"]
            }
            
            documentos.append(Document(page_content=texto, metadata=metadata))
    
    print(f"Total documentos cargados: {len(documentos)}")
    return documentos

def crear_chunks(documentos: list[Document]) -> list[Document]:
    """Divide documentos en fragmentos manejables."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = splitter.split_documents(documentos)
    print(f"Total chunks generados: {len(chunks)}")
    return chunks

def construir_base_conocimiento(chunks: list[Document]) -> Chroma:
    """Genera embeddings y almacena en ChromaDB."""
    print("Cargando modelo de embeddings...")
    embeddings = EmbeddingsLocales("all-MiniLM-L6-v2")
    
    print("Generando embeddings y construyendo base de conocimiento...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    print(f"Base de conocimiento construida en {CHROMA_DIR}")
    return vectorstore

def consultar(vectorstore: Chroma, pregunta: str, k: int = 4) -> None:
    """Consulta la base de conocimiento y muestra resultados."""
    print(f"\nPregunta: {pregunta}")
    print("-" * 50)
    
    resultados = vectorstore.similarity_search(pregunta, k=k)
    
    for i, doc in enumerate(resultados, 1):
        print(f"\nResultado {i}:")
        print(f"Marca: {doc.metadata['marca']}")
        print(f"Fuente: {doc.metadata['fuente']}")
        print(f"Fecha: {doc.metadata['fecha']}")
        print(f"Texto: {doc.page_content[:300]}...")

def main():
    # Paso 1: cargar artículos
    documentos = cargar_articulos()
    
    # Paso 2: crear chunks
    chunks = crear_chunks(documentos)
    
    # Paso 3: construir base de conocimiento
    vectorstore = construir_base_conocimiento(chunks)
    
    # Paso 4: pruebas de consulta
    preguntas = [
        "¿Qué movimientos financieros ha hecho Falabella recientemente?",
        "¿Qué estrategias de moda está siguiendo Studio F?",
        "¿Cómo está posicionando Zara sus colecciones en Colombia?"
    ]
    
    for pregunta in preguntas:
        consultar(vectorstore, pregunta)

if __name__ == "__main__":
    main()