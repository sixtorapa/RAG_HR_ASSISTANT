import os
import uuid
import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-testing")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from app import create_app, db as _db
from app.models import User, ChatSession, Message


class TestConfig:
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "test-secret-key"
    WTF_CSRF_ENABLED = False
    UP_VECTOR_STORE_PATH = "/tmp/test_vs"
    KNOWLEDGE_BASE_PATH = "/tmp/test_docs"
    UP_PROJECT_NAME = "Test HR KB"
    MODEL_NAME = "gpt-4o-mini"
    SYSTEM_INSTRUCTION = ""
    SQL_CONTEXT = ""
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


# ── KEY: clean the tables after every test ───────────────────
@pytest.fixture(scope="function", autouse=True)
def clean_db(app):
    yield
    with app.app_context():
        _db.session.rollback()
        Message.query.delete()
        ChatSession.query.delete()
        User.query.delete()
        _db.session.commit()


@pytest.fixture(scope="function")
def db(app):
    with app.app_context():
        yield _db


@pytest.fixture(scope="function")
def test_user(db):
    # Unique name per test, to avoid the UNIQUE constraint
    unique = uuid.uuid4().hex[:8]
    user = User(username=f"testuser_{unique}", email=f"test_{unique}@example.com", role="user")
    user.set_password("Password123!")
    db.session.add(user)
    db.session.commit()
    return user




@pytest.fixture(scope="function")
def test_chat_session(db, test_user):
    session = ChatSession(
        name="Test Chat",
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