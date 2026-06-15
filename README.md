# Retail Intelligence Colombia

Sistema de inteligencia competitiva para el mercado de retail de moda en Colombia.
Monitorea Falabella, Studio F y Zara Colombia a partir de fuentes públicas de noticias
y produce análisis accionables para equipos de mercadeo y estrategia.

## Arquitectura

```
Google News RSS
      │
      ▼
  main.py  ──────────────────►  data/raw/*.json
                                      │
                                      ▼
                              pipeline_rag.py  ──►  data/chroma_db/
                                                           │
                                                           ▼
                                                        app.py  (Streamlit UI)
```

- **`main.py`** — extrae artículos de Google News RSS por marca, aplica filtros de relevancia y antigüedad, guarda JSON en `data/raw/`.
- **`pipeline_rag.py`** — carga los JSON, divide en chunks, genera embeddings locales (`all-MiniLM-L6-v2`) y construye la base de conocimiento en ChromaDB.
- **`app.py`** — UI Streamlit con dos modos: reporte semanal con 6 dimensiones predefinidas + briefing ejecutivo adaptado por marca, y consulta libre.
- **`config.py`** — constantes compartidas (rutas, modelos).
- **`utils.py`** — código compartido: `EmbeddingsLocales`, `formatear_contexto`, `PROMPT_SISTEMA`.

## Requisitos

- Python 3.11
- [uv](https://docs.astral.sh/uv/) para gestión de dependencias
- Cuenta gratuita en [Groq](https://console.groq.com) para obtener una API key

## Instalación

```bash
uv sync
```

## Configuración

Crea el archivo `.streamlit/secrets.toml` con tu API key de Groq:

```toml
GROQ_API_KEY = "gsk_..."
```

O ingrésala directamente en la UI al iniciar la aplicación. Ver `.env.example` como referencia.

## Uso

### Primera vez

```bash
# 1. Extraer artículos de fuentes RSS
python main.py

# 2. Construir la base de conocimiento vectorial
python setup_db.py

# 3. Iniciar la UI
streamlit run app.py
```

### Actualizaciones de datos

Repite los pasos 1 y 2 para incorporar artículos nuevos. El paso 3 puede dejarse corriendo.

```bash
python main.py && python setup_db.py
```

## Datos

- `data/raw/` — artículos extraídos en JSON, organizados por marca y fecha de extracción.
- `data/chroma_db/` — base de conocimiento vectorial generada por `pipeline_rag.py`.

Ambos directorios están excluidos del repositorio (`.gitignore`). Los datos se generan localmente ejecutando el pipeline.

## Limitaciones

- Fuente única: Google News RSS (cobertura limitada a lo que indexa Google).
- Cobertura máxima: 2 años hacia atrás.
- Actualización manual: no hay scheduler automático.
- Filtros de relevancia basados en palabras clave, pueden producir falsos positivos o negativos.
