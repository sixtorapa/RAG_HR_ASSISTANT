import os
import uuid
import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-testing")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from app import create_app, db as _db
from app.models import User, Project, ChatSession, Message


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "test-secret-key"
    WTF_CSRF_ENABLED = False
    UP_VECTOR_STORE_PATH = "/tmp/test_vs"
    KNOWLEDGE_BASE_PATH = "/tmp/test_docs"
    UP_PROJECT_NAME = "Test HR KB"
    OPENAI_API_KEY = "sk-test-fake-key"
    USD_TO_EUR_RATE = 0.92
    HR_DB_URI = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def app():
    flask_app = create_app(TestConfig)
    with flask_app.app_context():
        _db.create_all()
        yield flask_app
        _db.drop_all()


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()


# ── CLAVE: limpiar tablas después de cada test ───────────────────
@pytest.fixture(scope="function", autouse=True)
def clean_db(app):
    yield
    with app.app_context():
        _db.session.rollback()
        Message.query.delete()
        ChatSession.query.delete()
        Project.query.delete()
        User.query.delete()
        _db.session.commit()


@pytest.fixture(scope="function")
def db(app):
    with app.app_context():
        yield _db


@pytest.fixture(scope="function")
def test_user(db):
    # Nombre único por test para evitar UNIQUE constraint
    unique = uuid.uuid4().hex[:8]
    user = User(username=f"testuser_{unique}", email=f"test_{unique}@example.com", role="user")
    user.set_password("Password123!")
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture(scope="function")
def test_project(db):
    unique = uuid.uuid4().hex[:8]
    project = Project(
        name=f"Test Project {unique}",
        document_path="/tmp/docs",
        vector_store_path=f"/tmp/vs_{unique}",
        status="READY",
        model_name="gpt-4o-mini",
        settings={},
        cost=0.0,
    )
    db.session.add(project)
    db.session.commit()
    return project


@pytest.fixture(scope="function")
def test_chat_session(db, test_user, test_project):
    session = ChatSession(
        name="Test Chat",
        project_id=test_project.id,
        user_id=test_user.id,
    )
    db.session.add(session)
    db.session.commit()
    return session


@pytest.fixture(scope="function")
def auth_client(app, client, test_user):
    with client.session_transaction() as sess:
        sess["_user_id"] = test_user.id
        sess["_fresh"] = True
    return client