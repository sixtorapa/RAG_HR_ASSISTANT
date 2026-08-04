# app/main/admin.py
"""Panel de actividad y exportación en CSV. Solo para administradores."""

from datetime import datetime, timedelta

from flask import Response, abort, redirect, render_template, request, stream_with_context, url_for
from flask_login import current_user, login_required

from app import db
from app.main import bp
from app.models import ChatSession, LoginSession, Message, User


@bp.route("/admin/activity")
@login_required
def admin_activity():
    if getattr(current_user, "role", None) != "admin":
        abort(403)


    # llevarte a una sesión válida para que el sidebar no rompa
    chat_sessions_list = (
        ChatSession.query
        .filter_by(user_id=current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    current_chat_session = chat_sessions_list[0] if chat_sessions_list else None

    if current_chat_session:
        return redirect(url_for("main.index", tab="activity", session=current_chat_session.id))

    # Si por lo que sea no hay sesiones, index ya crea una al entrar,
    # pero aquí le mandamos sin session
    return redirect(url_for("main.index", tab="activity"))
@bp.route("/admin/activity/export_sessions.csv")
@login_required
def export_sessions_csv():
    if getattr(current_user, "role", None) != "admin":
        abort(403)

    import csv
    import io

    def row(v):
        # Evita None y deja el CSV limpio
        return "" if v is None else v

    def dt(v):
        # Formato consistente y “humano” (sin milisegundos)
        return "" if v is None else v.strftime("%Y-%m-%d %H:%M:%S")

    @stream_with_context
    def generate():
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        # Cabecera
        writer.writerow([
            "login_session_id",
            "user_id",
            "username",
            "started_at",
            "last_activity_at",
            "ended_at",
            "duration_sec",
            "n_questions",
        ])
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)

        # Query histórico completo (streaming)
        q = (
            db.session.query(LoginSession, User)
            .join(User, User.id == LoginSession.user_id)
            .order_by(LoginSession.started_at.desc())
            .yield_per(1000)
        )

        for ls, u in q:
            writer.writerow([
                row(ls.id),
                row(u.id),
                row(u.username),
                dt(ls.started_at),
                dt(ls.last_activity_at),
                dt(ls.ended_at),
                row(ls.duration_sec),
                row(ls.n_questions),
            ])
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    filename = "detalles_sesiones_historico.csv"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return Response(generate(), mimetype="text/csv", headers=headers)
