# app/models.py
from app import db
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


# La tabla `project` se eliminó el 4-ago-2026. De sus ocho columnas, cinco eran
# una copia de variables de entorno (nombre, rutas, modelo), `status` valía
# siempre "READY", `id` solo existía para una clave foránea, y `cost` se
# acumulaba en una columna que no se mostraba en ninguna pantalla. Lo único con
# estado real —la instrucción de sistema y el contexto SQL— es configuración y
# vive en config.py.
#
# No era código de más: era una fuente de verdad duplicada. Cambiar
# UP_VECTOR_STORE_PATH no movía un proyecto ya creado, porque la ruta estaba
# congelada en la fila — uno de los tres fallos que solo aparecieron al
# desplegar en Lambda.


class ChatSession(db.Model):
    """
    Representa una conversación específica de un usuario.
    """
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False, default="Nuevo Chat")

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

    # Control de acceso por departamento (guardarril): lista de slugs de
    # knowledge_base/<department>/ a los que este usuario tiene acceso.
    # Semántica resuelta en get_allowed_departments(), no leer este campo directo.
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
        Resuelve el alcance de departamentos de este usuario para el prefiltro
        de seguridad en retrieval (qa_chain._build_scoped_retriever).

        Returns:
            None  -> sin restricción (solo admin). Ve todos los departamentos.
            []    -> sin acceso a ningún departamento (fail closed por defecto:
                     un usuario "user" sin allowed_departments asignado no ve nada).
            list  -> restringido exactamente a esos departamentos.
        """
        if self.role == "admin":
            return None
        return list(self.allowed_departments) if self.allowed_departments else []

    def __repr__(self):
        return f'<User {self.username}>'
