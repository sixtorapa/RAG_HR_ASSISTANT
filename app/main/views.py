# app/main/views.py
"""HTML screens: home, chat, activity."""

from datetime import datetime, timedelta

from flask import current_app, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app import db
from app.main import bp
from app.models import ChatSession, LoginSession, Message, User


@bp.route("/")
@login_required
def index():
    """Single screen: home, chat and activity."""
    tab = (request.args.get("tab") or "home").lower().strip()
    if tab not in ("home", "chat", "activity"):
        tab = "home"

    # Non-admins are not allowed into the activity view
    if tab == "activity" and getattr(current_user, "role", None) != "admin":
        tab = "home"


    session_id = (request.args.get("session") or "").strip()
    session = None

    # If a session id is given, accept it only if it belongs to the user
    if session_id:
        session = ChatSession.query.filter_by(
            id=session_id, user_id=current_user.id,
        ).first()

    # The user's sessions, for the sidebar
    sessions_list = (
        ChatSession.query
        .filter_by(user_id=current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )

    # With no valid session_id, fall back to the user's most recent one
    if session is None:
        session = sessions_list[0] if sessions_list else None

    # If there is none, create one. add+commit is required to avoid a detached instance
    if session is None:
        session = ChatSession(name="New chat", user_id=current_user.id)
        db.session.add(session)
        db.session.commit()
        sessions_list = [session]  # so it shows up in the sidebar straight away

    # Avoid lazy-loading session.messages, which raises DetachedInstanceError
    messages = (
        Message.query
        .filter_by(session_id=session.id, user_id=current_user.id)
        .order_by(Message.timestamp.asc())
        .all()
    )


    activity_sessions = []
    admin_activity = None  # summary + series for the dashboard (admin only)

    if tab == "activity" and getattr(current_user, "role", None) == "admin":
        # Detail table: the last 200 sessions
        activity_sessions = (
            db.session.query(LoginSession, User)
            .join(User, User.id == LoginSession.user_id)
            .order_by(LoginSession.started_at.desc())
            .limit(200)
            .all()
        )

        # Dashboard summary: the last 30 days
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

        # Series: sessions per day
        by_day = {}
        for (ls, _) in last_30:
            key = (ls.started_at.date().isoformat() if ls.started_at else None)
            if not key:
                continue
            by_day[key] = by_day.get(key, 0) + 1

        days = sorted(by_day.keys())
        sessions_per_day = [by_day[d] for d in days]

        # Ranking: top users by questions and sessions
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
        kb_name=current_app.config["UP_PROJECT_NAME"],
        model_name=current_app.config["MODEL_NAME"],
        session=session,
        messages=messages,
        sessions=sessions_list,
        active_tab=tab,
        activity_sessions=activity_sessions,
        admin_activity=admin_activity,  
    )
@bp.route("/chat/<session_id>")
@login_required
def chat_session(session_id):
    # Legacy route: redirect to the single shell
    return redirect(url_for("main.index", tab="chat", session=session_id))
@bp.route("/check_status")
def check_status():
    """Knowledge-base status, polled by the UI."""
    return jsonify({"status": "READY", "name": current_app.config["UP_PROJECT_NAME"]})
