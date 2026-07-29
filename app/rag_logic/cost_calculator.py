# app/rag_logic/cost_calculator.py
"""Coste por consulta a partir del uso de tokens."""

import logging
from datetime import date

logger = logging.getLogger(__name__)

PRICES_LAST_UPDATED = date(2025, 1, 1)   # ← actualizar al revisar precios

# Precios en USD por 1 millón de tokens.
# Fuente: https://openai.com/es-ES/api/pricing/
TOKEN_PRICES = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o":      {"prompt": 2.50, "completion": 10.00},
}


def _usd_to_eur_rate() -> float:
    """
    Tipo de cambio desde la config de Flask.
    Devuelve 1.0 si no hay contexto de aplicación o si no está configurado,
    avisando por log de que el importe resultante está en USD, no en EUR.
    """
    try:
        from flask import current_app
        rate = current_app.config.get("USD_TO_EUR_RATE")
    except RuntimeError:            # llamada fuera del contexto de aplicación
        rate = None

    if rate is None:
        logger.warning(
            "USD_TO_EUR_RATE no configurado: el coste se devuelve en USD, no en EUR."
        )
        return 1.0
    return rate


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Coste de una llamada al LLM, en EUR si USD_TO_EUR_RATE está configurado
    (en USD si no lo está; se avisa por log).

    Devuelve 0.0 si el modelo no tiene precio conocido, registrando un error:
    quien llama hace `project.cost += calculate_cost(...)`, así que un fallo
    contable no debe romper la respuesta al usuario.
    """
    prices = TOKEN_PRICES.get(model_name)
    if prices is None:
        logger.error(
            "Modelo '%s' sin precio en TOKEN_PRICES: esta llamada se contabiliza "
            "como 0 y el total del proyecto queda infravalorado.",
            model_name,
        )
        return 0.0

    if (date.today() - PRICES_LAST_UPDATED).days > 180:
        logger.warning("TOKEN_PRICES sin revisar desde %s.", PRICES_LAST_UPDATED)

    prompt_cost     = (prompt_tokens / 1_000_000) * prices["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * prices["completion"]

    return (prompt_cost + completion_cost) * _usd_to_eur_rate()
