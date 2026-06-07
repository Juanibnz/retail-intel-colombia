# Corre este script una vez antes de desplegar
# para verificar que todo funciona localmente

from pipeline_rag import cargar_articulos, crear_chunks, construir_base_conocimiento

print("Construyendo base de conocimiento...")
documentos = cargar_articulos()
chunks = crear_chunks(documentos)
vectorstore = construir_base_conocimiento(chunks)
print("Listo.")