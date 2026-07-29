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

from app import create_app  # noqa: E402  (después de fijar las rutas de caché)
from config import Config  # noqa: E402

app = create_app(Config)

# binary_support=True para que las respuestas no textuales (imágenes, ficheros)
# viajen en base64 en vez de corromperse.
handler = make_lambda_handler(app, binary_support=True)
