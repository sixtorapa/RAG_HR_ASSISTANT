# app/main/guards.py
"""
Input guardrails for /ask.

Both cut BEFORE the first LLM call and BEFORE anything is persisted: once the
data reaches the model or our own database, it has left the perimeter. They live
in their own module because they are the compliance surface of the system and it
should be visible in the file tree.
"""

import os
from datetime import datetime

from flask import jsonify
from flask_login import current_user

from app.models import Message
from app.rag_logic.pii_guard import find_sensitive_entities


def _quota_block():
    """
    Daily question cap for the whole installation.

    It exists because the public demo runs against Bedrock, which is pay-per-use
    with no automatic ceiling: AWS budget alarms warn, they do not stop. OpenAI is
    prepaid and stops itself, so Railway does not need this — hence it only
    activates when DAILY_QUESTION_LIMIT is set and above zero.

    The counter lives in the DATABASE, not in process memory: in Lambda each
    container has its own memory, so a RAM counter would apply the cap per
    container and N parallel containers would multiply it.
    """
    limit = int(os.environ.get("DAILY_QUESTION_LIMIT", "0") or 0)
    if limit <= 0:
        return None

    today = datetime.utcnow().date()
    used_today = Message.query.filter(
        Message.sender == "user",
        Message.timestamp >= datetime.combine(today, datetime.min.time()),
    ).count()
    if used_today < limit:
        return None

    print(f"🚫 QUOTA: {used_today}/{limit} questions used today — blocked user={current_user.id}")
    return jsonify({
        "error": (
            f"This demo is capped at {limit} questions per day to control "
            f"inference cost, and today's quota is used up. "
            f"Please try again tomorrow."
        ),
        "quota_used": used_today,
        "quota_limit": limit,
    }), 429
def _dlp_block(text, context):
    """Input guardrail. Returns the error response, or None when clean."""
    findings = find_sensitive_entities(text)
    if not findings:
        return None
    print(f"🚫 DLP: blocked {context} — detected: {sorted({f.kind for f in findings})}")
    return jsonify({
        "error": (
            "Your message appears to contain identifying or financial data "
            "(IBAN, card or ID document). For security it is neither processed "
            "nor stored. Please rephrase the question without those details."
        )
    }), 400
