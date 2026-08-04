# app/main/auth.py
"""Inicio y cierre de sesión, y el registro de sesiones de login."""

from datetime import datetime

from flask import flash, redirect, render_template, request, url_for
from flask import session as flask_session
from flask_login import current_user, login_required, login_user, logout_user

from app import db
from app.main import bp
from app.models import LoginSession, User


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        
        user = User.query.filter_by(username=username, is_active=True).first()
        if user and user.check_password(password):
            login_user(user)

            # Auditoría básica (sin IP)
            user.last_login = datetime.utcnow()

            # Cerrar sesión anterior abierta si existiera (por seguridad/consistencia)
            prev = (
                LoginSession.query
                .filter_by(user_id=user.id, ended_at=None)
                .order_by(LoginSession.started_at.desc())
                .first()
            )
            if prev:
                now = datetime.utcnow()
                prev.ended_at = now
                prev.duration_sec = max(0, int((now - prev.started_at).total_seconds()))

            # Crear nueva sesión de login
            ls = LoginSession(
                user_id=user.id,
                started_at=datetime.utcnow(),
                n_questions=0,
                last_activity_at=datetime.utcnow(),
            )
            db.session.add(ls)
            db.session.commit()

            flask_session["login_session_id"] = ls.id

            return redirect(url_for("main.index"))



        flash("Credenciales incorrectas", "danger")

    return render_template("login.html")
@bp.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    try:
        ls_id = flask_session.get("login_session_id")
        if ls_id:
            ls = LoginSession.query.filter_by(id=ls_id, user_id=current_user.id).first()
            if ls and ls.ended_at is None:
                now = datetime.utcnow()
                ls.ended_at = now
                ls.duration_sec = max(0, int((now - ls.started_at).total_seconds()))
                db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass

    flask_session.pop("login_session_id", None)

    logout_user()
    return redirect(url_for("main.login"))
def _bump_login_session_question() -> None:
    """
    Incrementa n_questions de la sesión de login activa (si existe).
    No rompe si falta (p.ej. sesión antigua / key inexistente).
    """
    try:
        ls_id = flask_session.get("login_session_id")
        if not ls_id:
            return

        ls = LoginSession.query.filter_by(id=ls_id, user_id=current_user.id).first()
        if not ls:
            return

        ls.n_questions = (ls.n_questions or 0) + 1
        ls.last_activity_at = datetime.utcnow()
        db.session.commit()
    except Exception:
        # Silencioso: esto nunca debe tumbar /ask
        try:
            db.session.rollback()
        except Exception:
            pass
