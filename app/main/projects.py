# app/main/projects.py
"""Alta y mantenimiento del proyecto único y de las sesiones de chat."""

import os

from flask import current_app, flash, jsonify, redirect, request, url_for
from flask_login import current_user, login_required

from app import db
from app.main import bp
from app.models import ChatSession, Message, Project
from app.rag_logic.ingester import process_and_store_documents
from app.rag_logic.qa_chain import chain_cache


def _get_or_create_single_project() -> Project:
    """
    SINGLE PROJECT legacy-safe:
    Prioridad de selección:
      1) Project con vector_store_path == UP_VECTOR_STORE_PATH (si existe)
      2) Project con name == HR Knowledge Base (evita UNIQUE name al crear)
      3) Primer Project existente
      4) Si no hay ninguno, crear uno nuevo
    """
    cfg = current_app.config

    name = (cfg.get("UP_PROJECT_NAME") or "HR Knowledge Base").strip()
    doc_path = (cfg.get("KNOWLEDGE_BASE_PATH") or "").strip()
    vector_path = (cfg.get("UP_VECTOR_STORE_PATH") or "").strip()

    target_vs = (vector_path or "").replace("\\", "/").strip()

    project = None

    # 1) Buscar por vector_store_path
    for p in Project.query.all():
        p_vs = (p.vector_store_path or "").replace("\\", "/").strip()
        if p_vs == target_vs and target_vs:
            project = p
            break

    # 2) Buscar por name
    if project is None and name:
        project = Project.query.filter_by(name=name).first()

    # 3) Primer project
    if project is None:
        project = Project.query.first()

    # 4) Crear
    if project is None:
        project = Project(
            name=name,
            document_path=doc_path or "-",
            vector_store_path=vector_path,
            status="READY",
            model_name="gpt-4o",
            settings={},
        )
        db.session.add(project)
        db.session.commit()
        return project

    # Normalización segura
    dirty = False

    if name and project.name != name:
        project.name = name
        dirty = True

    if doc_path and project.document_path != doc_path:
        project.document_path = doc_path
        dirty = True

    if project.status != "READY":
        project.status = "READY"
        dirty = True

    if project.settings is None:
        project.settings = {}
        dirty = True

    # Actualizar vector_store_path SOLO si no rompe UNIQUE
    cur_vs = (project.vector_store_path or "").replace("\\", "/").strip()
    if target_vs and cur_vs != target_vs:
        conflict = None
        for p in Project.query.all():
            if p.id == project.id:
                continue
            p_vs = (p.vector_store_path or "").replace("\\", "/").strip()
            if p_vs == target_vs:
                conflict = p
                break
        if conflict is None:
            project.vector_store_path = vector_path
            dirty = True

    if dirty:
        db.session.commit()

    return project
def _clear_chain_cache_for_project(project_id: str) -> None:
    """En qa_chain.py la key del cache es: f"{project_id}::{model_name}" """
    try:
        prefix = f"{project_id}::"
        for k in list(chain_cache.keys()):
            if str(k).startswith(prefix):
                del chain_cache[k]
    except Exception:
        pass
@bp.route("/reindex", methods=["POST"])
def reindex():
    """
    Reindexa el único vector store:
    KNOWLEDGE_BASE_PATH -> UP_VECTOR_STORE_PATH (por defecto: vector_store/info)
    """
    # Protección simple: token por header o querystring
    token_cfg = (current_app.config.get("UP_ADMIN_TOKEN") or "").strip()
    token_in = (request.headers.get("X-UP-ADMIN-TOKEN") or request.args.get("token") or "").strip()
    if token_cfg and token_in != token_cfg:
        return ("Forbidden", 403)

    cfg = current_app.config
    doc_path = (cfg.get("KNOWLEDGE_BASE_PATH") or "").strip()
    vector_path = (cfg.get("UP_VECTOR_STORE_PATH") or "").strip()

    if not doc_path:
        flash("KNOWLEDGE_BASE_PATH is empty. Revisa config/.env.", "danger")
        return redirect(url_for("main.index"))

    if not os.path.exists(doc_path):
        flash(f"KNOWLEDGE_BASE_PATH no existe: {doc_path}", "danger")
        return redirect(url_for("main.index"))

    try:
        os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    except Exception as e:
        flash(f"No se pudo crear la carpeta de vector_store: {e}", "danger")
        return redirect(url_for("main.index"))

    try:
        ok = process_and_store_documents(doc_path, vector_path)
    except Exception as e:
        flash(f"Error reindexando: {e}", "danger")
        return redirect(url_for("main.index"))

    try:
        chain_cache.clear()
    except Exception:
        pass

    if ok:
        flash("✅ Índice actualizado correctamente.", "success")
    else:
        flash("⚠️ No se indexaron documentos (carpeta vacía o sin texto).", "warning")

    return redirect(url_for("main.index"))
@bp.route("/create_project", methods=["POST"])
def create_project():
    flash("Esta app ya no usa proyectos. Usa Home / Asistente UP.", "warning")
    return redirect(url_for("main.index"))
@bp.route("/delete_project/<project_id>", methods=["POST"])
def delete_project(project_id):
    flash("Esta app ya no permite borrar proyectos. (SINGLE PROJECT).", "warning")
    return redirect(url_for("main.index"))
@bp.route("/edit_project/<project_id>", methods=["POST"])
def edit_project(project_id):
    flash("Esta app ya no permite editar proyectos. (SINGLE PROJECT).", "warning")
    return redirect(url_for("main.index"))
@bp.route("/create_chat", methods=["POST"])
@login_required
def create_chat():
    project = _get_or_create_single_project()
    new_session = ChatSession(name="Nuevo chat", project=project, user_id=current_user.id)

    db.session.add(new_session)
    db.session.commit()
    return redirect(url_for("main.index", tab="chat", session=new_session.id))
@bp.route("/delete_chat/<session_id>", methods=["POST"])
@login_required
def delete_chat(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    project_id = session.project_id

    Message.query.filter_by(session_id=session.id, user_id=current_user.id).delete(synchronize_session=False)


    db.session.delete(session)
    db.session.commit()
    _clear_chain_cache_for_project(project_id)

    flash("Chat eliminado.", "success")
    return redirect(url_for("main.index", tab="chat"))
