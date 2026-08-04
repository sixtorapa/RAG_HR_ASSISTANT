# app/main/routes.py
"""
The chat endpoints. Nothing else.

The pipeline lives in pipeline.py, the guardrails in guards.py, and the remaining
screens in views.py, chats.py, auth.py and admin.py. This file once ran to 1,590
lines doing all six jobs at once.
"""

import os

from flask import current_app, flash, jsonify, redirect, request, url_for
from flask_login import current_user, login_required

from app import db
from app.main import bp
from app.main.guards import _dlp_block, _quota_block
from app.main.auth import _bump_login_session_question
from app.main.pipeline import _answer_question, _finalize, _ToolBox
from app.models import ChatSession, Message
from app.rag_logic.console_logger import ConsoleLogger


@bp.route("/health")
def health():
    """
    Liveness probe. Deliberately without @login_required: a healthcheck from
    Docker, a load balancer or Lambda cannot log in.

    Deliberately cheap too — it confirms the process is alive and serving, and
    checks neither the database nor the vector store. A healthcheck that depends
    on external services turns a brief database blip into a container restart,
    which makes the incident worse rather than better.
    """
    return {
        "status": "ok",
        "llm_provider": os.environ.get("LLM_PROVIDER", "openai"),
    }, 200
@bp.route("/ask/<session_id>", methods=["POST"])
@login_required
def ask(session_id):
    """
    Answer a question in a chat session.

    The endpoint is only a doorman: it validates, applies both guardrails and
    delegates. The whole pipeline lives in _answer_question(), shared with
    edit_and_resubmit().
    _answer_question(), compartido con edit_and_resubmit().
    """
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()

    payload = request.get_json(silent=True) or {}
    question_text = payload.get("question")
    if not question_text:
        return jsonify({"error": "Falta la question."}), 400

    model_name = payload.get("model_name") or current_app.config["MODEL_NAME"]

    # Both guardrails run BEFORE the LLM and BEFORE anything is persisted: once
    # the data reaches the model or our own database, it has left the perimeter.
    blocked = _quota_block() or _dlp_block(question_text, f"user={current_user.id} session={session_id}")
    if blocked:
        return blocked

    _bump_login_session_question()

    try:
        box = _ToolBox(
            model_name, current_user, ConsoleLogger(),
            use_web_search=payload.get("use_web_search", False),
        )

        final_result, titulo = _answer_question(
            session, question_text, model_name, box,
        )

        return _finalize(session, question_text, final_result, title_question=titulo)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error del servidor: {str(e)}"}), 500
@bp.route("/edit_and_resubmit/<int:message_id>", methods=["POST"])
@login_required
def edit_and_resubmit(message_id):
    """
    Rewrite an already-sent question and regenerate the answer.

    This function used to repeat the entire ask() pipeline — router, dispatch
    loop, formatting, persistence — around 270 duplicated lines. That copy-paste
    produced the three `NameError: memory_store` crashes ruff exposed when F821
    was re-enabled: someone duplicated the block and dropped a line.
    Ahora las dos rutas comparten _answer_question().
    """
    user_message = Message.query.filter_by(id=message_id, user_id=current_user.id).first_or_404()
    session = user_message.session

    new_text = (request.json or {}).get("new_question")
    if not new_text or user_message.sender != "user":
        return jsonify({"error": "Inválido"}), 400

    blocked = _dlp_block(new_text, f"user={current_user.id} message={message_id}")
    if blocked:
        return blocked

    model_name = current_app.config["MODEL_NAME"]

    try:
        # Everything after this message is deleted — only the current user's —
        # and its content rewritten. The history the router sees is what came
        # before this point, not what existed when it was first asked.
        posteriores = Message.query.filter(
            Message.session_id == session.id,
            Message.user_id == current_user.id,
            Message.timestamp > user_message.timestamp,
        ).all()
        for m in posteriores:
            db.session.delete(m)
        user_message.content = new_text

        box = _ToolBox(model_name, current_user, ConsoleLogger())

        final_result, _ = _answer_question(
            session, new_text, model_name, box,
            historial_hasta=user_message.timestamp,
        )

        return _finalize(
            session, new_text, final_result,
            existing_user_message=user_message,
        )

    except Exception as e:
        db.session.rollback()
        print(f"Error regen: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/clear_history/<session_id>", methods=["POST"])
@login_required
def clear_history(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()

    try:
        session.messages.delete()
        db.session.commit()
        flash("Historial borrado.", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Error: {e}", "danger")

    return redirect(url_for("main.chat_session", session_id=session_id))
