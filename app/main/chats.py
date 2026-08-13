# app/main/chats.py
"""Chat sessions and corpus re-indexing."""

import os

from flask import current_app, flash, jsonify, redirect, request, url_for
from flask_login import current_user, login_required

from app import db
from app.main import bp
from app.models import ChatSession, Message
from app.rag_logic.ingester import process_and_store_documents
from app.rag_logic.qa_chain import chain_cache


def _clear_chain_cache() -> None:
    """
    Empty the chain cache. Called after re-indexing: the cache key does not
    change on re-ingestion, so without this the app would keep serving chains
    pointing at the previous index until the process restarts.
    """
    try:
        chain_cache.clear()
    except Exception:
        pass


@bp.route("/reindex", methods=["POST"])
def reindex():
    """
    Re-index the single vector store:
    KNOWLEDGE_BASE_PATH -> UP_VECTOR_STORE_PATH (por defecto: vector_store/info)
    """
    # Simple protection: token via header or querystring
    token_cfg = (current_app.config.get("UP_ADMIN_TOKEN") or "").strip()
    token_in = (request.headers.get("X-UP-ADMIN-TOKEN") or request.args.get("token") or "").strip()
    if token_cfg and token_in != token_cfg:
        return ("Forbidden", 403)

    cfg = current_app.config
    doc_path = (cfg.get("KNOWLEDGE_BASE_PATH") or "").strip()
    vector_path = (cfg.get("UP_VECTOR_STORE_PATH") or "").strip()

    if not doc_path:
        flash("KNOWLEDGE_BASE_PATH is empty. Check config/.env.", "danger")
        return redirect(url_for("main.index"))

    if not os.path.exists(doc_path):
        flash(f"KNOWLEDGE_BASE_PATH does not exist: {doc_path}", "danger")
        return redirect(url_for("main.index"))

    try:
        os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    except Exception as e:
        flash(f"Could not create the vector_store folder: {e}", "danger")
        return redirect(url_for("main.index"))

    try:
        ok = process_and_store_documents(doc_path, vector_path)
    except Exception as e:
        flash(f"Error while re-indexing: {e}", "danger")
        return redirect(url_for("main.index"))

    _clear_chain_cache()

    if ok:
        flash("✅ Index updated successfully.", "success")
    else:
        flash("⚠️ No documents were indexed (empty folder, or no extractable text).", "warning")

    return redirect(url_for("main.index"))


@bp.route("/create_chat", methods=["POST"])
@login_required
def create_chat():
    new_session = ChatSession(name="New chat", user_id=current_user.id)
    db.session.add(new_session)
    db.session.commit()
    return redirect(url_for("main.index", tab="chat", session=new_session.id))


@bp.route("/delete_chat/<session_id>", methods=["POST"])
@login_required
def delete_chat(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()

    Message.query.filter_by(session_id=session.id, user_id=current_user.id).delete(synchronize_session=False)
    db.session.delete(session)
    db.session.commit()

    flash("Chat deleted.", "success")
    return redirect(url_for("main.index", tab="chat"))
