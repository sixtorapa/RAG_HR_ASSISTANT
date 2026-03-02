"""
test_models.py — Tests de los modelos SQLAlchemy.

Por qué importan:
  - Validan constraints (unique, nullable) antes de que fallen en producción
  - Verifican las relaciones entre modelos (cascade delete, backref)
  - Son el contrato entre la app y la DB
"""

import pytest
from datetime import datetime
from app.models import User, Project, ChatSession, Message


class TestUserModel:
    """Tests del modelo User y su lógica de contraseñas."""

    def test_create_user(self, db):
        user = User(username="alice", email="alice@example.com")
        user.set_password("SecurePass123!")
        db.session.add(user)
        db.session.commit()

        fetched = User.query.filter_by(username="alice").first()
        assert fetched is not None
        assert fetched.email == "alice@example.com"
        assert fetched.role == "user"  # default
        assert fetched.is_active is True

    def test_password_hashing(self, db):
        """La contraseña nunca se guarda en texto plano."""
        user = User(username="bob")
        user.set_password("MyPassword!")
        db.session.add(user)
        db.session.commit()

        assert user.password_hash != "MyPassword!"
        assert user.check_password("MyPassword!") is True
        assert user.check_password("WrongPassword") is False

    def test_username_unique_constraint(self, db):
        """Dos usuarios con el mismo username deben fallar."""
        u1 = User(username="duplicate")
        u1.set_password("pass1")
        u2 = User(username="duplicate")
        u2.set_password("pass2")

        db.session.add(u1)
        db.session.commit()
        db.session.add(u2)

        with pytest.raises(Exception):  # IntegrityError
            db.session.commit()
        db.session.rollback()

    def test_admin_role(self, db):
        admin = User(username="admin_user", role="admin")
        admin.set_password("Admin123!")
        db.session.add(admin)
        db.session.commit()

        assert admin.role == "admin"

    def test_user_repr(self, db, test_user):
        assert "testuser" in repr(test_user)

    def test_uuid_primary_key(self, db, test_user):
        """El ID debe ser un UUID válido, no un int."""
        assert len(test_user.id) == 36
        assert test_user.id.count("-") == 4


class TestProjectModel:
    """Tests del modelo Project."""

    def test_create_project(self, db, test_project):
        assert test_project.id is not None
        assert test_project.status == "READY"
        assert test_project.cost == 0.0

    def test_project_repr(self, db, test_project):
        assert "Test Project" in repr(test_project)

    def test_project_status_pending_by_default(self, db):
        p = Project(
            name="Pending Project",
            document_path="/tmp/d",
            vector_store_path="/tmp/vs2",
        )
        db.session.add(p)
        db.session.commit()
        assert p.status == "PENDING"

    def test_project_cost_accumulation(self, db, test_project):
        """Simula acumulación de coste como hace la app real."""
        test_project.cost += 0.0023
        test_project.cost += 0.0041
        db.session.commit()

        refreshed = Project.query.get(test_project.id)
        assert abs(refreshed.cost - 0.0064) < 0.0001

    def test_project_settings_json(self, db, test_project):
        """El campo settings debe soportar dicts anidados."""
        test_project.settings = {
            "search_k": 20,
            "reranker": "flashrank",
            "hybrid_weights": {"bm25": 0.55, "vector": 0.45},
        }
        db.session.commit()

        refreshed = Project.query.get(test_project.id)
        assert refreshed.settings["hybrid_weights"]["bm25"] == 0.55


class TestChatSessionModel:
    """Tests de la sesión de chat y sus mensajes."""

    def test_create_session(self, db, test_chat_session):
        assert test_chat_session.id is not None
        assert test_chat_session.name == "Test Chat"

    def test_session_default_name(self, db, test_user, test_project):
        session = ChatSession(
            project_id=test_project.id,
            user_id=test_user.id,
        )
        db.session.add(session)
        db.session.commit()
        assert session.name == "Nuevo Chat"

    def test_add_messages_to_session(self, db, test_chat_session, test_user):
        user_msg = Message(
            session_id=test_chat_session.id,
            user_id=test_user.id,
            sender="user",
            content="¿Cuál es la política de vacaciones?",
        )
        bot_msg = Message(
            session_id=test_chat_session.id,
            user_id=test_user.id,
            sender="bot",
            content="La política de vacaciones establece 23 días anuales.",
            sources=[{"file": "hr_policy.pdf", "page": 3}],
        )
        db.session.add_all([user_msg, bot_msg])
        db.session.commit()

        messages = Message.query.filter_by(session_id=test_chat_session.id).all()
        assert len(messages) == 2

    def test_message_sources_json(self, db, test_chat_session, test_user):
        """Las fuentes se guardan como JSON y se recuperan correctamente."""
        sources = [
            {"file": "policy.pdf", "page": 1, "score": 0.87},
            {"file": "handbook.docx", "page": 5, "score": 0.72},
        ]
        msg = Message(
            session_id=test_chat_session.id,
            user_id=test_user.id,
            sender="bot",
            content="Respuesta con fuentes",
            sources=sources,
        )
        db.session.add(msg)
        db.session.commit()

        fetched = Message.query.get(msg.id)
        assert fetched.sources[0]["score"] == 0.87
        assert len(fetched.sources) == 2

    def test_cascade_delete_session_deletes_messages(self, db, test_user, test_project):
        """Al borrar una sesión, sus mensajes se eliminan (cascade)."""
        session = ChatSession(
            name="Temp",
            project_id=test_project.id,
            user_id=test_user.id,
        )
        db.session.add(session)
        db.session.commit()

        msg = Message(
            session_id=session.id,
            user_id=test_user.id,
            sender="user",
            content="Pregunta que desaparecerá",
        )
        db.session.add(msg)
        db.session.commit()
        msg_id = msg.id

        db.session.delete(session)
        db.session.commit()

        assert Message.query.get(msg_id) is None

    def test_message_sender_values(self, db, test_chat_session, test_user):
        """Solo 'user' y 'bot' son valores válidos para sender."""
        for sender in ("user", "bot"):
            msg = Message(
                session_id=test_chat_session.id,
                user_id=test_user.id,
                sender=sender,
                content=f"Mensaje de {sender}",
            )
            db.session.add(msg)
        db.session.commit()

        msgs = Message.query.filter_by(session_id=test_chat_session.id).all()
        senders = {m.sender for m in msgs}
        assert senders == {"user", "bot"}