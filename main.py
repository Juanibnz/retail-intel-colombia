import json
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

FUENTES = {
    "falabella": [
        "https://news.google.com/rss/search?q=Falabella+Colombia+retail&hl=es-419&gl=CO&ceid=CO:es",
        "https://news.google.com/rss/search?q=Falabella+tienda+Colombia&hl=es-419&gl=CO&ceid=CO:es",
        "https://news.google.com/rss/search?q=Falabella+moda+Colombia&hl=es-419&gl=CO&ceid=CO:es",
    ],
    "studio_f": [
        "https://news.google.com/rss/search?q=%22Studio+F%22+Colombia&hl=es-419&gl=CO&ceid=CO:es",
        "https://news.google.com/rss/search?q=%22Studio+F%22+moda+mujer&hl=es-419&gl=CO&ceid=CO:es",
    ],
    "zara_colombia": [
        "https://news.google.com/rss/search?q=Zara+Colombia+moda&hl=es-419&gl=CO&ceid=CO:es",
        "https://news.google.com/rss/search?q=Zara+Inditex+Colombia&hl=es-419&gl=CO&ceid=CO:es",
    ],
}

FILTROS = {
    "falabella": {
    "incluir": ["Colombia", "colombiano", "Bogotá", "Medellín", "Cali", 
                "tienda", "retail", "moda", "ropa", "temporada", 
                "descuento", "oferta", "colección"],
    "excluir": ["banco", "crédito", "hipotecario", "viajes", "vuelo", 
                "hotel", "Chile", "Perú", "Argentina", "México",
                "televisor", "televisión", "electrodoméstico", "lavadora",
                "nevera", "celular", "smartphone", "computador", "laptop",
                "tecnología", "hogar", "mueble", "colchón", "jardín"]
    },
    "studio_f": {
        "incluir": ["Colombia", "colombiano", "Bogotá", "moda", "colección",
                    "temporada", "tienda", "mujer"],
        "excluir": [],
    },
    "zara_colombia": {
        "incluir": ["Colombia", "colombiano", "Bogotá", "moda", "colección",
                    "Zara", "Inditex", "tienda"],
        "excluir": [],
    },
}

DIAS_MAXIMOS = 730
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------

def limpiar_html(texto: str) -> str:
    if not texto:
        return ""
    soup = BeautifulSoup(texto, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def extraer_articulos(marca: str, url: str) -> list[dict]:
    feed = feedparser.parse(url)
    articulos = []

    for entry in feed.entries:
        articulo = {
            "marca": marca,
            "titulo": entry.get("title", ""),
            "fuente": entry.get("source", {}).get("title", "Google News"),
            "url": entry.get("link", ""),
            "fecha": entry.get("published", ""),
            "resumen": limpiar_html(entry.get("summary", "")),
            "extraido_en": datetime.now().isoformat(),
        }
        articulos.append(articulo)

    return articulos


def extraer_todos(marca: str, urls: list[str]) -> list[dict]:
    todos = []
    for url in urls:
        todos.extend(extraer_articulos(marca, url))

    vistos = set()
    unicos = []
    for a in todos:
        if a["url"] not in vistos:
            vistos.add(a["url"])
            unicos.append(a)

    print(f"  → {len(unicos)} artículos únicos antes de filtrar")
    return unicos

# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def es_relevante(articulo: dict, marca: str) -> bool:
    texto = f"{articulo['titulo']} {articulo['resumen']}".lower()
    filtro = FILTROS[marca]

    tiene_inclusion = any(
        palabra.lower() in texto
        for palabra in filtro["incluir"]
    )

    tiene_exclusion = any(
        palabra.lower() in texto
        for palabra in filtro["excluir"]
    )

    return tiene_inclusion and not tiene_exclusion


def es_reciente(articulo: dict, dias: int = DIAS_MAXIMOS) -> bool:
    fecha_str = articulo.get("fecha", "")
    if not fecha_str:
        return False

    try:
        fecha = parsedate_to_datetime(fecha_str)
        fecha_naive = fecha.replace(tzinfo=None)
        limite = datetime.now() - timedelta(days=dias)
        return fecha_naive >= limite
    except Exception:
        return False


def filtrar_articulos(articulos: list[dict], marca: str) -> list[dict]:
    relevantes = [a for a in articulos if es_relevante(a, marca)]
    recientes = [a for a in relevantes if es_reciente(a)]

    descartados_relevancia = len(articulos) - len(relevantes)
    descartados_tiempo = len(relevantes) - len(recientes)

    print(f"  → {len(recientes)} válidos")
    print(f"  → {descartados_relevancia} descartados por relevancia")
    print(f"  → {descartados_tiempo} descartados por antigüedad")

    return recientes

# ---------------------------------------------------------------------------
# Almacenamiento
# ---------------------------------------------------------------------------

def guardar_articulos(articulos: list[dict], marca: str):
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    archivo = OUTPUT_DIR / f"{marca}_{fecha_hoy}.json"

    with open(archivo, "w", encoding="utf-8") as f:
        json.dump(articulos, f, ensure_ascii=False, indent=2)

    print(f"{marca}: {len(articulos)} artículos guardados en {archivo}")

# ---------------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------------

def main():
    for marca, urls in FUENTES.items():
        print(f"\nProcesando {marca}...")
        articulos = extraer_todos(marca, urls)
        articulos_filtrados = filtrar_articulos(articulos, marca)
        guardar_articulos(articulos_filtrados, marca)


if __name__ == "__main__":
    main()
