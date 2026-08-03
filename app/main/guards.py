# app/main/guards.py
"""
Guardarraíles de entrada de /ask.

Los dos cortan ANTES de la primera llamada al LLM y ANTES de persistir nada: si
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
    Tope diario de preguntas para toda la instalación.

    Existe porque la demo pública va contra Bedrock, que es pago por uso SIN tope
    automático: las alarmas de presupuesto de AWS avisan, no cortan. OpenAI es
    prepago y se frena solo, así que Railway no lo necesita — por eso solo se
    activa si DAILY_QUESTION_LIMIT está definida y es > 0.

    El contador vive en la BASE DE DATOS, no en memoria del proceso: en Lambda
    cada contenedor tiene su propia memoria, así que un contador en RAM aplicaría
    el tope "por contenedor" y N contenedores en paralelo lo multiplicarían.
    """
    limite = int(os.environ.get("DAILY_QUESTION_LIMIT", "0") or 0)
    if limite <= 0:
        return None

    hoy = datetime.utcnow().date()
    usadas = Message.query.filter(
        Message.sender == "user",
        Message.timestamp >= datetime.combine(hoy, datetime.min.time()),
    ).count()
    if usadas < limite:
        return None

    print(f"🚫 CUOTA: {usadas}/{limite} preguntas usadas hoy — bloqueada user={current_user.id}")
    return jsonify({
        "error": (
            f"Esta demo tiene un límite de {limite} preguntas al día para "
            f"controlar el coste de inferencia, y ya se han consumido. "
            f"Vuelve a intentarlo mañana."
        ),
        "quota_used": usadas,
        "quota_limit": limite,
    }), 429
def _dlp_block(texto, contexto):
    """Guardarraíl de entrada. Devuelve la respuesta de error, o None si está limpio."""
    hallazgos = find_sensitive_entities(texto)
    if not hallazgos:
        return None
    print(f"🚫 DLP: bloqueado {contexto} — detectado: {sorted({f.kind for f in hallazgos})}")
    return jsonify({
        "error": (
            "Tu mensaje parece contener datos identificativos o financieros "
            "(IBAN, tarjeta o documento de identidad). Por seguridad, no se procesa "
            "ni se guarda. Reformula la pregunta sin incluir esos datos."
        )
    }), 400
