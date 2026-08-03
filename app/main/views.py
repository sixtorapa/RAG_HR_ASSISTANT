# app/main/views.py
"""Pantallas HTML: home, chat, ajustes."""

from datetime import datetime, timedelta

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app import db
from app.main import bp
from app.main.projects import _clear_chain_cache_for_project, _get_or_create_single_project
from app.models import ChatSession, LoginSession, Message, Project, User
from app.rag_logic.prompt_templates import get_all_templates, get_template_prompt


@bp.route("/")
@login_required
def index():
    """Pantalla única (Home + Asistente UP) sin gestión de proyectos."""
    project = _get_or_create_single_project()

    tab = (request.args.get("tab") or "home").lower().strip()
    if tab not in ("home", "chat", "activity"):
        tab = "home"

    # Si no es admin, activity no está permitido
    if tab == "activity" and getattr(current_user, "role", None) != "admin":
        tab = "home"


    session_id = (request.args.get("session") or "").strip()
    session = None

    # Si viene session id, solo si es del usuario
    if session_id:
        session = ChatSession.query.filter_by(
            id=session_id,
            project_id=project.id,
            user_id=current_user.id,
        ).first()

    # Lista de sesiones del usuario (para sidebar tipo ChatGPT)
    sessions_list = (
        ChatSession.query
        .filter_by(project_id=project.id, user_id=current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )

    # Si no viene session_id válido, usamos la última del usuario
    if session is None:
        session = sessions_list[0] if sessions_list else None

    # Si no hay ninguna, crear una nueva (IMPORTANTE: add+commit para evitar Detached)
    if session is None:
        session = ChatSession(
            name="Nuevo chat",
            project=project,
            user_id=current_user.id,
        )
        db.session.add(session)
        db.session.commit()
        sessions_list = [session]  # opcional: para que aparezca inmediatamente

    # ✅ Evitar lazy-load sobre session.messages (evita DetachedInstanceError)
    messages = (
        Message.query
        .filter_by(session_id=session.id, user_id=current_user.id)
        .order_by(Message.timestamp.asc())
        .all()
    )


    activity_sessions = []
    admin_activity = None  # <-- resumen + series para dashboard (solo admin)

    if tab == "activity" and getattr(current_user, "role", None) == "admin":
        # Tabla (detalle): últimas 200 sesiones
        activity_sessions = (
            db.session.query(LoginSession, User)
            .join(User, User.id == LoginSession.user_id)
            .order_by(LoginSession.started_at.desc())
            .limit(200)
            .all()
        )

        # Dashboard (resumen): últimos 30 días
        since = datetime.utcnow() - timedelta(days=30)

        last_30 = (
            db.session.query(LoginSession, User)
            .join(User, User.id == LoginSession.user_id)
            .filter(LoginSession.started_at >= since)
            .order_by(LoginSession.started_at.asc())
            .all()
        )

        total_sessions_30d = len(last_30)
        active_users_30d = len({u.id for (_, u) in last_30}) if last_30 else 0
        total_questions_30d = sum(int(ls.n_questions or 0) for (ls, _) in last_30) if last_30 else 0

        durations = [int(ls.duration_sec) for (ls, _) in last_30 if ls.duration_sec is not None]
        avg_duration_sec_30d = int(sum(durations) / len(durations)) if durations else 0

        # Series: sesiones por día
        by_day = {}
        for (ls, _) in last_30:
            key = (ls.started_at.date().isoformat() if ls.started_at else None)
            if not key:
                continue
            by_day[key] = by_day.get(key, 0) + 1

        days = sorted(by_day.keys())
        sessions_per_day = [by_day[d] for d in days]

        # Ranking: top usuarios por preguntas (y sesiones)
        per_user = {}
        for (ls, u) in last_30:
            item = per_user.get(u.username) or {"sessions": 0, "questions": 0}
            item["sessions"] += 1
            item["questions"] += int(ls.n_questions or 0)
            per_user[u.username] = item

        top_by_questions = sorted(
            [{"username": k, **v} for k, v in per_user.items()],
            key=lambda x: (x["questions"], x["sessions"]),
            reverse=True,
        )[:8]

        admin_activity = {
            "kpis": {
                "total_sessions_30d": total_sessions_30d,
                "active_users_30d": active_users_30d,
                "total_questions_30d": total_questions_30d,
                "avg_duration_sec_30d": avg_duration_sec_30d,
            },
            "series": {
                "days": days,
                "sessions_per_day": sessions_per_day,
            },
            "top_users": top_by_questions,
        }

    return render_template(
        "dashboard.html",
        project=project,
        session=session,
        messages=messages,
        sessions=sessions_list,
        active_tab=tab,
        activity_sessions=activity_sessions,
        admin_activity=admin_activity,  # <-- NUEVO
    )
@bp.route("/project/<project_id>")
@login_required
def project_overview(project_id):
    project = Project.query.get_or_404(project_id)
    sessions = (
        ChatSession.query
        .filter_by(project_id=project.id, user_id=current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return render_template("project_overview.html", project=project, sessions=sessions)
@bp.route("/chat/<session_id>")
@login_required
def chat_session(session_id):
    # Legacy: redirige al shell único
    return redirect(url_for("main.index", tab="chat", session=session_id))
@bp.route("/settings/<project_id>", methods=["GET", "POST"])
@login_required
def project_settings(project_id):
    project = Project.query.get_or_404(project_id)

    if request.method == "POST":
        current_settings = project.settings or {}
        current_settings["k_value"] = int(request.form.get("k_value", 5))
        tpl = request.form.get("template_selected", "")

        if tpl and tpl != "custom":
            current_settings["system_instruction"] = get_template_prompt(tpl)
            current_settings["template_type"] = tpl
        else:
            custom = (request.form.get("system_instruction", "") or "").strip()
            current_settings["system_instruction"] = custom
            current_settings["template_type"] = "custom" if custom else "generico"

        current_settings["sql_context"] = request.form.get("sql_context", "")
        project.settings = current_settings
        db.session.commit()

        _clear_chain_cache_for_project(project.id)


        flash("Ajustes guardados.", "success")
        return redirect(url_for("main.project_settings", project_id=project.id))

    settings = project.settings or {}
    return render_template(
        "settings.html",
        project=project,
        settings=settings,
        templates=get_all_templates(),
        current_template=settings.get("template_type", "generico"),
        current_instruction=settings.get("system_instruction", ""),
    )
@bp.route("/check_status")
def check_status():
    projects = Project.query.all()
    statuses = {p.id: p.status for p in projects}
    return jsonify(statuses)
