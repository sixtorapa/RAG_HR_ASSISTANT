# app/main/__init__.py
from flask import Blueprint

bp = Blueprint("main", __name__)

# Los módulos se importan al final para que `bp` ya exista cuando registran sus
# rutas. Cada uno cubre una responsabilidad; routes.py es solo el chat.
from app.main import auth, chats, views, admin, routes  # noqa: E402,F401
