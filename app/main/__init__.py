# app/main/__init__.py
from flask import Blueprint

bp = Blueprint("main", __name__)

# Modules are imported last so that `bp` already exists when they register their
# routes. Each module covers one responsibility; routes.py is only the chat.
from app.main import auth, chats, views, admin, routes  # noqa: E402,F401
