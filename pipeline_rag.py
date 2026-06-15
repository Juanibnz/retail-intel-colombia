import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL
from utils import EmbeddingsLocales


def cargar_articulos() -> list[Document]:
    """Carga todos los JSON y los convierte en Documents de LangChain."""
    documentos = []

    for archivo in DATA_DIR.glob("*.json"):
        with open(archivo, "r", encoding="utf-8") as f:
            articulos = json.load(f)

        for articulo in articulos:
            texto = f"""
Marca: {articulo['marca']}
Título: {articulo['titulo']}
Fecha: {articulo['fecha']}
Fuente: {articulo['fuente']}

{articulo['resumen']}
            """.strip()

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
    embeddings = EmbeddingsLocales(EMBEDDING_MODEL)

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
    documentos = cargar_articulos()
    chunks = crear_chunks(documentos)
    vectorstore = construir_base_conocimiento(chunks)

    preguntas = [
        "¿Qué movimientos financieros ha hecho Falabella recientemente?",
        "¿Qué estrategias de moda está siguiendo Studio F?",
        "¿Cómo está posicionando Zara sus colecciones en Colombia?"
    ]

    for pregunta in preguntas:
        consultar(vectorstore, pregunta)


if __name__ == "__main__":
    main()
