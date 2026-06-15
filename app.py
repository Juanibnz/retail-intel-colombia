import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time

from config import CHROMA_DIR, EMBEDDING_MODEL, GROQ_MODEL
from utils import EmbeddingsLocales, formatear_contexto, PROMPT_SISTEMA


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


@st.cache_resource
def cargar_recursos(groq_key: str):
    embeddings = EmbeddingsLocales(EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=groq_key
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

with st.sidebar:
    st.header("Configuración")
    groq_key = st.text_input(
        "API key de Groq",
        type="password",
        help="Obtén una key gratuita en console.groq.com"
    )
    st.caption("Tu key no se almacena ni se comparte.")

    st.divider()
    st.markdown("**¿Para qué marca trabajas?**")
    marca_usuario = st.selectbox(
        "Tu marca",
        options=["", "Falabella", "Studio F", "Zara Colombia"],
        help="El briefing se adaptará a tu perspectiva"
    )

    st.divider()
    st.markdown("**¿Qué es esto?**")
    st.caption(
        "Sistema de inteligencia de mercado que monitorea "
        "Falabella, Studio F y Zara Colombia usando fuentes "
        "públicas de noticias procesadas con LLMs."
    )
    st.markdown("**Limitaciones actuales**")
    st.caption(
        "Fuentes: Google News RSS · "
        "Cobertura: últimos 2 años · "
        "Actualización: manual"
    )

if not groq_key or not marca_usuario:
    if not groq_key:
        st.info("Ingresa tu API key de Groq en el panel izquierdo para continuar.")
    if not marca_usuario:
        st.info("Selecciona tu marca en el panel izquierdo para continuar.")
    st.stop()


def inicializar_si_necesario():
    """Construye la base de conocimiento si no existe."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        st.info("Inicializando base de conocimiento por primera vez. Esto toma unos minutos...")

        from main import main as extraer
        from pipeline_rag import cargar_articulos, crear_chunks, construir_base_conocimiento

        with st.spinner("Extrayendo artículos de fuentes públicas..."):
            extraer()

        with st.spinner("Construyendo base de conocimiento vectorial..."):
            documentos = cargar_articulos()
            chunks = crear_chunks(documentos)
            construir_base_conocimiento(chunks)

        st.success("Base de conocimiento lista.")
        st.rerun()


inicializar_si_necesario()

vectorstore, llm = cargar_recursos(groq_key)
cadena = construir_cadena(vectorstore, llm)

tab_reporte, tab_consulta = st.tabs(["📊 Reporte semanal", "💬 Consulta libre"])

with tab_reporte:
    st.subheader("Reporte de inteligencia semanal")

    with st.expander("⚡ Briefing ejecutivo", expanded=True):
        if st.button("Generar briefing", type="primary"):
            with st.spinner("Analizando..."):
                prompt_briefing = f"""
Estás trabajando para el equipo estratégico de {marca_usuario} en Colombia.

Tu trabajo es producir un briefing comparativo que le diga al director
de estrategia exactamente dónde está {marca_usuario} en relación a sus
competidores en este momento.

REGLA CRÍTICA: Cada afirmación debe incluir fuente y fecha entre paréntesis.
Formato: (Fuente · Fecha)
Si no tienes fuente para una afirmación, no la incluyas.

Estructura exacta:

⚡ POSICIÓN COMPETITIVA DE {marca_usuario.upper()} ESTA SEMANA
[Una oración que describe dónde está {marca_usuario} en el tablero
competitivo ahora mismo. ¿Está adelante, atrás, o en movimiento?]

📊 TABLERO COMPARATIVO
[Tabla de 3 filas y 3 columnas]
| Dimensión | {marca_usuario} | Competidores |
|-----------|-----------------|--------------|
| Expansión | [estado actual con fuente] | [quién se mueve y cómo] |
| Digital   | [estado actual con fuente] | [quién se mueve y cómo] |
| Producto  | [estado actual con fuente] | [quién se mueve y cómo] |

🔴 DONDE {marca_usuario.upper()} ESTÁ PERDIENDO TERRENO
[Una oración concreta. La dimensión donde un competidor específico
está avanzando más rápido. (Fuente · Fecha)]

🟢 DONDE {marca_usuario.upper()} TIENE VENTAJA
[Una oración concreta. La dimensión donde {marca_usuario} lleva
la delantera ahora mismo. (Fuente · Fecha)]

❓ LA DECISIÓN QUE NO PUEDE ESPERAR
[Una sola pregunta estratégica que surge directamente de la
comparación. Que incomode. Que obligue a actuar.]

Nada más. Sin introducciones ni conclusiones adicionales.
"""
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}
                )

                competidores = [m for m in ["Falabella", "Studio F", "Zara Colombia"]
                               if m != marca_usuario]

                docs_marca = retriever.invoke(
                    f"estrategia expansión digital producto {marca_usuario} Colombia"
                )
                docs_comp1 = retriever.invoke(
                    f"estrategia expansión digital producto {competidores[0]} Colombia"
                )
                docs_comp2 = retriever.invoke(
                    f"estrategia expansión digital producto {competidores[1]} Colombia"
                )

                todos_los_docs = []
                urls_vistas = set()
                for doc in docs_marca + docs_comp1 + docs_comp2:
                    url = doc.metadata.get("url", "")
                    if url not in urls_vistas:
                        urls_vistas.add(url)
                        todos_los_docs.append(doc)

                contexto_completo = formatear_contexto(todos_los_docs)

                prompt_final = ChatPromptTemplate.from_messages([
                    ("system", PROMPT_SISTEMA.replace("{contexto}", contexto_completo)
                                             .replace("{fecha_hoy}",
                                                      datetime.now().strftime("%d de %B de %Y"))),
                    ("human", prompt_briefing)
                ])

                chain_briefing = prompt_final | llm | StrOutputParser()
                briefing = chain_briefing.invoke({})
                st.markdown(briefing)

                st.divider()
                st.caption("**Fuentes consultadas:**")
                fuentes_vistas = set()
                for doc in todos_los_docs:
                    fuente_key = f"{doc.metadata['fuente']}|{doc.metadata['fecha']}"
                    if fuente_key not in fuentes_vistas:
                        fuentes_vistas.add(fuente_key)
                        col1, col2, col3 = st.columns([3, 2, 2])
                        with col1:
                            st.caption(doc.metadata['titulo'][:60] + "...")
                        with col2:
                            st.caption(doc.metadata['fuente'])
                        with col3:
                            st.caption(doc.metadata['fecha'][:16])

    st.divider()
    st.caption("Análisis automatizado sobre las 6 dimensiones clave del mercado")

    if st.button("Generar reporte completo", type="primary"):
        for i, seccion in enumerate(PREGUNTAS_REPORTE):
            with st.expander(f"**{seccion['titulo']}**", expanded=True):
                with st.spinner("Analizando fuentes..."):
                    respuesta = cadena.invoke(seccion["pregunta"])
                    st.markdown(respuesta)
                    if i < len(PREGUNTAS_REPORTE) - 1:
                        time.sleep(8)

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
