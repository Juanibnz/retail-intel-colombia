# CLAUDE.md

## Proyecto

Sistema de inteligencia competitiva para retail de moda en Colombia. Monitorea
Falabella, Studio F y Zara Colombia via Google News RSS. Pipeline RAG con
ChromaDB y UI Streamlit.

## Comandos

```bash
# Extraer artículos de fuentes RSS
python main.py

# Construir / reconstruir la base de conocimiento vectorial
python setup_db.py

# Iniciar la UI
streamlit run app.py
```

## Arquitectura

- `main.py` — extractor RSS (feedparser + BeautifulSoup). Escribe JSONs en `data/raw/`.
- `pipeline_rag.py` — ingesta RAG. Lee JSONs, crea chunks, construye ChromaDB en `data/chroma_db/`.
- `app.py` — UI Streamlit. Punto de entrada principal. Reporte semanal + consulta libre.
- `setup_db.py` — script de una sola ejecución que llama a `pipeline_rag.py`.
- `config.py` — constantes compartidas (rutas, nombres de modelos).
- `utils.py` — código compartido: `EmbeddingsLocales`, `formatear_contexto`, `PROMPT_SISTEMA`.

## Configuración

API key de Groq se ingresa en la UI en tiempo de ejecución.
Alternativa: `.streamlit/secrets.toml` con `GROQ_API_KEY = "..."` (excluido de git).

## Modelo de embeddings

`all-MiniLM-L6-v2` (sentence-transformers, local). Constante canónica: `EMBEDDING_MODEL` en `config.py`.
No cambiar sin reconstruir la ChromaDB completa (`python setup_db.py`).

## Dependencias

Gestionadas con `uv`. Fuente de verdad: `pyproject.toml`.
