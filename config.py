# config.py
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv()


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "change-me-in-production"

    # ── SQLAlchemy: users, chat sessions, messages ─────────────────────────
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL")
        or "sqlite:///" + os.path.join(basedir, "app.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── HR analytics database (SQLite here; swap via env for Postgres) ─────
    HR_DB_URI = os.environ.get(
        "HR_DB_URI",
        "sqlite:///" + os.path.join(basedir, "hr_data.db"),
    )

    # ── Vector store ────────────────────────────────────────────────────────
    UP_VECTOR_STORE_PATH = os.environ.get(
        "UP_VECTOR_STORE_PATH",
        os.path.join(basedir, "vector_store", "info"),
    )

    # ── Document folder ─────────────────────────────────────────────────────
    KNOWLEDGE_BASE_PATH = os.environ.get(
        "KNOWLEDGE_BASE_PATH",
        os.path.join(basedir, "docs"),
    )

    # ── Identity and model ──────────────────────────────────────────────────
    UP_PROJECT_NAME = os.environ.get("UP_PROJECT_NAME", "HR Knowledge Base")
    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

    # ── Assistant settings ──────────────────────────────────────────────────
    # These lived in a JSON column of the `project` table, which was no more than
    # a copy of the configuration, able to drift from it. Here they are what they
    # always were: configuration.
    #   SYSTEM_INSTRUCTION -> prepended to the document answer prompt
    #   SQL_CONTEXT        -> schema description for the SQL generator
    SYSTEM_INSTRUCTION = os.environ.get("SYSTEM_INSTRUCTION", "")
    SQL_CONTEXT = os.environ.get("SQL_CONTEXT", "")
    UP_ADMIN_TOKEN = os.environ.get("UP_ADMIN_TOKEN", "")

    # ── LLM ─────────────────────────────────────────────────────────────────
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    # ── LangSmith observability (optional — set to enable) ──────────────────
    LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "hr-kb-assistant")