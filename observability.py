# observability.py
"""
LangSmith observability setup.

Call `init_langsmith()` once at application startup (e.g. inside create_app()).
If the required env vars are not set, it silently skips — no crash in dev.

Usage in create_app():
    from app.observability import init_langsmith
    init_langsmith()

What you get in LangSmith:
  - Every LangChain chain / LLM call traced automatically
  - Tool calls + routing decisions visible per request
  - Latency, token usage, cost per trace
  - Session grouping via LANGCHAIN_PROJECT
"""

import os
import logging

logger = logging.getLogger(__name__)


def init_langsmith() -> bool:
    """
    Activate LangSmith tracing by setting the required env vars.
    Returns True if tracing was enabled, False otherwise.
    """
    tracing  = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower()
    api_key  = os.environ.get("LANGCHAIN_API_KEY", "")
    project  = os.environ.get("LANGCHAIN_PROJECT", "hr-kb-assistant")

    if tracing != "true" or not api_key:
        logger.info("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable).")
        return False

    # LangChain reads these env vars automatically — no extra code needed.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = api_key
    os.environ["LANGCHAIN_PROJECT"]    = project

    logger.info(f"✅ LangSmith tracing enabled — project: '{project}'")
    return True


def get_run_metadata(user_id: str = "", session_id: str = "") -> dict:
    """
    Return metadata dict to pass as `config={'metadata': ...}` to LangChain chains.
    This enriches traces in LangSmith with user / session context.

    Example:
        chain.invoke(payload, config={"metadata": get_run_metadata(user_id="u_42")})
    """
    return {
        "project": os.environ.get("LANGCHAIN_PROJECT", "hr-kb-assistant"),
        "user_id": user_id,
        "session_id": session_id,
        "environment": os.environ.get("FLASK_ENV", "development"),
    }
