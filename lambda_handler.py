"""
Punto de entrada para AWS Lambda.

Flask habla WSGI: una función síncrona que recibe (environ, start_response).
Lambda no habla WSGI: entrega un diccionario JSON con el evento de API Gateway
y espera otro diccionario de vuelta. `apig_wsgi` es el traductor entre ambos.

Por qué apig-wsgi y no Mangum: Mangum es para ASGI (FastAPI, Starlette).
Flask es WSGI. Meter Mangum aquí sería el adaptador equivocado.

Todo lo de nivel de módulo se ejecuta UNA vez por contenedor, en la fase de
init del arranque en frío, y se reutiliza en las invocaciones calientes.
Por eso `create_app()` va aquí fuera y no dentro de `handler`: construirlo en
cada invocación pagaría el arranque siempre.
"""

import os

from apig_wsgi import make_lambda_handler

# En Lambda el único directorio escribible es /tmp. El vector store viaja
# horneado en la imagen y se lee en solo-lectura; cualquier cosa que necesite
# escribir (caché de librerías, ficheros temporales) tiene que ir a /tmp.
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface")

# ── El vector store NO puede servirse desde la imagen ────────────────────────
# La receta habitual dice "hornea el vector store en la imagen y léelo en
# solo lectura". Con Chroma no funciona: su SQLite se abre en lectura-ESCRITURA
# incluso para consultar, porque necesita su journal. El síntoma es
#   "attempt to write a readonly database (code: 8)"
# y la consulta falla aunque el índice esté ahí.
#
# Se copia a /tmp en la fase de init: 8,7 MB, una vez por contenedor, no por
# invocación. /tmp da 512 MB y es efímero, pero el índice es de solo consulta y
# viaja en la imagen, así que reconstruirlo no cuesta nada.
_ORIGEN = "/var/task/vector_store"
_DESTINO = "/tmp/vector_store"
if os.path.isdir(_ORIGEN) and not os.path.isdir(_DESTINO):
    import shutil

    shutil.copytree(_ORIGEN, _DESTINO)
if os.path.isdir(_DESTINO):
    os.environ["UP_VECTOR_STORE_PATH"] = os.path.join(_DESTINO, "info")

from app import create_app  # noqa: E402  (después de fijar rutas y cachés)
from config import Config  # noqa: E402

app = create_app(Config)

# Flask sitúa `instance_path` junto al código, o sea dentro de /var/task, que en
# Lambda es de SOLO LECTURA. `ChatMemoryStore` guarda ahí la memoria conversacional
# como un Chroma persistente, así que sin esto /ask revienta con 500 al intentar
# crear los ficheros. Se reubica en /tmp, el único directorio escribible.
#
# Consecuencia asumida y que hay que saber explicar: /tmp es POR CONTENEDOR y
# efímero. La memoria conversacional sobrevive entre invocaciones calientes del
# mismo contenedor, pero no se comparte entre contenedores ni sobrevive a un
# arranque en frío. Para memoria duradera habría que sacarla a un almacén externo
# (la BD, S3 o EFS) — es la misma limitación que ya tiene `chain_cache`.
app.instance_path = "/tmp/instance"
os.makedirs(app.instance_path, exist_ok=True)

# binary_support=True para que las respuestas no textuales (imágenes, ficheros)
# viajen en base64 en vez de corromperse.
handler = make_lambda_handler(app, binary_support=True)
