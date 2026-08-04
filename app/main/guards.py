# app/main/guards.py
"""
Guardarraíles de entrada de /ask.

Los dos cortan ANTES de la primera call al LLM y ANTES de persistir nada: si
el dato llega al modelo o a nuestra propia base de datos, ya ha salido del
perímetro. Viven en su propio módulo porque son la parte de compliance del
sistema y conviene que se vean.
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

    print(f"🚫 CUOTA: {used_today}/{limit} preguntas used_today today — bloqueada user={current_user.id}")
    return jsonify({
        "error": (
            f"Esta demo tiene un límite de {limit} preguntas al día para "
            f"controlar el cost de inferencia, y ya se han consumido. "
            f"Vuelve a intentarlo mañana."
        ),
        "quota_used": used_today,
        "quota_limit": limit,
    }), 429
def _dlp_block(text, context):
    """Input guardrail. Returns the error response, or None when clean."""
    findings = find_sensitive_entities(text)
    if not findings:
        return None
    print(f"🚫 DLP: bloqueado {context} — detectado: {sorted({f.kind for f in findings})}")
    return jsonify({
        "error": (
            "Tu mensaje parece contener datos identificativos o financieros "
            "(IBAN, tarjeta o documento de identidad). Por seguridad, no se procesa "
            "ni se guarda. Reformula la question sin incluir esos datos."
        )
    }), 400
