# app/models.py
from app import db
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


class Project(db.Model):
    """
    Representa la Base de Conocimiento (La carpeta indexada).
    """
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), index=True, unique=True, nullable=False)
    document_path = db.Column(db.String(255), nullable=False)
    vector_store_path = db.Column(db.String(255), unique=True, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='PENDING')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Configuraciones globales del proyecto
    model_name = db.Column(db.String(50), nullable=True)
    settings = db.Column(db.JSON, nullable=True)
    cost = db.Column(db.Float, nullable=False, default=0.0)

    # Relación: Un Proyecto tiene muchas sesiones de chat
    sessions = db.relationship('ChatSession', backref='project', lazy='dynamic', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Project {self.name}>'


class ChatSession(db.Model):
    """
    Representa una conversación específica de un usuario.
    """
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False, default="Nuevo Chat")

    project_id = db.Column(db.String(36), db.ForeignKey('project.id'), nullable=False)
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
    Sesión de login del usuario (observación básica).
    1 fila por login. Se cierra en logout (o queda abierta si no hubo logout).
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

    def __repr__(self):
        return f'<User {self.username}>'
