"""
AWS Lambda entry point.

Flask speaks WSGI: a synchronous function taking (environ, start_response).
Lambda does not: it hands over a JSON dictionary holding the API Gateway event
and expects another back. `apig_wsgi` is the translator between the two.

Why apig-wsgi and not Mangum: Mangum is for ASGI (FastAPI, Starlette). Flask is
WSGI, so Mangum here would be the wrong adapter.

Everything at module level runs ONCE per container, during the cold-start init
phase, and is reused by warm invocations. That is why `create_app()` sits out
here rather than inside `handler`: building it per invocation would pay the
startup cost every time.
"""

import os

from apig_wsgi import make_lambda_handler

# In Lambda the only writable directory is /tmp. The vector store travels baked
# into the image and is read from there; anything that needs to write — library
# caches, temporary files — has to go to /tmp.
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface")

# ── The vector store cannot be served from the image ─────────────────────────
# The usual advice is "bake the vector store into the image and read it in
# place". That does not work with Chroma: its SQLite opens read-WRITE even for
# queries, because it needs its journal. The symptom is
#   "attempt to write a readonly database (code: 8)"
# and retrieval fails while the index sits right there.
#
# It is copied to /tmp during the init phase: 8.7 MB, once per container rather
# than once per invocation. /tmp gives 512 MB and is ephemeral, but the index is
# read-only and travels in the image, so rebuilding it costs nothing.
_ORIGEN = "/var/task/vector_store"
_DESTINO = "/tmp/vector_store"
if os.path.isdir(_ORIGEN) and not os.path.isdir(_DESTINO):
    import shutil

    shutil.copytree(_ORIGEN, _DESTINO)
if os.path.isdir(_DESTINO):
    os.environ["UP_VECTOR_STORE_PATH"] = os.path.join(_DESTINO, "info")

from app import create_app  # noqa: E402  (after paths and caches are set)
from config import Config  # noqa: E402

app = create_app(Config)

# Flask puts `instance_path` next to the code, inside /var/task, which is
# READ-ONLY in Lambda. Anything writing there fails with a 500, so it is moved to
# /tmp, the only writable directory.
#
# The consequence is worth knowing: /tmp is PER CONTAINER and ephemeral. Whatever
# is written there survives warm invocations of the same container, but is not
# shared between containers and does not survive a cold start.
app.instance_path = "/tmp/instance"
os.makedirs(app.instance_path, exist_ok=True)

# binary_support=True so non-text responses (images, files) travel base64-encoded
# instead of being corrupted.
handler = make_lambda_handler(app, binary_support=True)
