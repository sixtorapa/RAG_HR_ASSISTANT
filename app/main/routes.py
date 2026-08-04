# app/main/routes.py
"""
Los endpoints del chat. Nada más.

El pipeline vive en pipeline.py, los guardarraíles en guards.py, y el resto de
pantallas en views.py / projects.py / auth.py / admin.py. Este fichero llegó a
tener 1.590 líneas haciendo las seis cosas a la vez.
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
    Liveness probe. SIN @login_required a propósito: un healthcheck de Docker,
    de un balanceador o de Lambda no puede iniciar sesión.

    Deliberadamente barato: solo confirma que el proceso está vivo y sirviendo.
    NO comprueba la base de datos ni el vector store. Un healthcheck que
    depende de servicios externos convierte una caída momentánea de la BD en
    un reinicio del contenedor, y eso empeora el incidente en vez de arreglarlo.
    """
    return {
        "status": "ok",
        "llm_provider": os.environ.get("LLM_PROVIDER", "openai"),
    }, 200
@bp.route("/ask/<session_id>", methods=["POST"])
@login_required
def ask(session_id):
    """
    Responder una question en una sesión de chat.

    El endpoint solo hace de portero: valida, aplica los dos guardarraíles y
    delega. Todo el pipeline —router, herramientas, formato— vive en
    _answer_question(), compartido con edit_and_resubmit().
    """
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()

    payload = request.get_json(silent=True) or {}
    question_text = payload.get("question")
    if not question_text:
        return jsonify({"error": "Falta la question."}), 400

    model_name = payload.get("model_name") or current_app.config["MODEL_NAME"]

    # Los dos guardarraíles van ANTES del LLM y ANTES de persistir nada: si el
    # dato llega al modelo o a nuestra propia BD, ya ha salido del perímetro.
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
    Reescribir una question ya enviada y regenerar la respuesta.

    Antes esta función repetía el pipeline entero de ask() —router, bucle de
    herramientas, formato, persistencia—, unas 270 líneas calcadas. De ese
    copia-pega salieron los tres `NameError: memory_store` que destapó ruff al
    reactivar F821: alguien duplicó el bloque y se dejó una línea.
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
        # Se borra todo lo posterior a este mensaje —solo del usuario actual— y
        # se reescribe su contenido. El historial que verá el router es el
        # anterior a este punto, no el que había cuando se preguntó la primera vez.
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
