# app/models.py
from app import db
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


# The `project` table was removed. Of its eight columns, five were a copy of
# environment variables (name, paths, model), `status` was always "READY", `id`
# existed only for a foreign key, and `cost` accumulated in a column no screen
# displayed. The only real state — the system instruction and the SQL context —
# is configuration and lives in config.py.
#
# The cost was never the extra code: it was a duplicated source of truth.
# Changing UP_VECTOR_STORE_PATH did not move an existing project, because the
# path was frozen in the row — one of the three failures that only appeared on
# Lambda.


class ChatSession(db.Model):
    """
    A single user conversation.
    """
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False, default="New chat")

    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relación: Una sesión tiene muchos mensajes
    messages = db.relationship(
        'Message',
        backref='session',
        lazy='dynamic',
        cascade="all, delete-orphan"
    )


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    session_id = db.Column(db.String(36), db.ForeignKey('chat_session.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)

    sender = db.Column(db.String(10), nullable=False)  # 'user' o 'bot'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    sources = db.Column(db.JSON, nullable=True)

    user = db.relationship('User')

    def __repr__(self):
        return f'<Message {self.id} from {self.sender}>'

    

from flask_login import UserMixin


class LoginSession(db.Model):
    """
    A user login session, for basic activity tracking.

    One row per login, closed on logout (and left open when there is none).
    """
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.String(36), db.ForeignKey("user.id"), nullable=False, index=True)

    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    ended_at = db.Column(db.DateTime, nullable=True)

    duration_sec = db.Column(db.Integer, nullable=True)  # se calcula al cerrar
    n_questions = db.Column(db.Integer, default=0, nullable=False)

    last_activity_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User")

    def __repr__(self):
        return f"<LoginSession {self.id} user={self.user_id} started_at={self.started_at}>"



class User(UserMixin, db.Model):

    """
    Usuario autenticado del sistema (tipo ChatGPT).
    """
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=True, index=True)

    password_hash = db.Column(db.String(255), nullable=False)

    role = db.Column(db.String(20), default="user")  # user | admin
    is_active = db.Column(db.Boolean, default=True)

    # Department access control: the knowledge_base/<department>/ slugs this user
    # may see. The semantics are resolved in get_allowed_departments(); do not
    # read this field directly.
    allowed_departments = db.Column(db.JSON, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # Relación: un usuario tiene muchos chats
    sessions = db.relationship(
        'ChatSession',
        backref='user',
        lazy='dynamic',
        cascade="all, delete-orphan"
    )

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def get_allowed_departments(self):
        """
        Resolve this user's department scope for the retrieval security filter.

        Returns:
            None  -> unrestricted, admin only. Sees every department.
            []    -> no access to any department. Fail-closed by default: a
                     regular user with no departments assigned sees nothing.
            list  -> restricted to exactly those departments.
        """
        if self.role == "admin":
            return None
        return list(self.allowed_departments) if self.allowed_departments else []

    def __repr__(self):
        return f'<User {self.username}>'
