# app/rag_logic/cost_calculator.py
"""Per-query cost from token usage."""

import logging
from datetime import date

logger = logging.getLogger(__name__)

PRICES_LAST_UPDATED = date(2025, 1, 1)   # ← actualizar al revisar precios

# Prices in USD per million tokens.
# Source: https://openai.com/api/pricing/
TOKEN_PRICES = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o":      {"prompt": 2.50, "completion": 10.00},
}


def _usd_to_eur_rate() -> float:
    """
    Exchange rate from the Flask config.

    Returns 1.0 when there is no application context or no rate configured,
    warning in the log that the amount is in USD rather than EUR.
    """
    try:
        from flask import current_app
        rate = current_app.config.get("USD_TO_EUR_RATE")
    except RuntimeError:            # called outside the application context
        rate = None

    if rate is None:
        logger.warning(
            "USD_TO_EUR_RATE not configured: cost returned in USD, not EUR."
        )
        return 1.0
    return rate


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Cost of one LLM call, in EUR when USD_TO_EUR_RATE is configured and in USD
    when it is not — with a warning in either case.

    Returns 0.0 for a model with no known price, logging an error: the caller adds
    this to the request cost, so an accounting failure must not break the answer.
    """
    prices = TOKEN_PRICES.get(model_name)
    if prices is None:
        logger.error(
            "Model '%s' has no price in TOKEN_PRICES: this call counts as zero "
            "and the total is under-reported.",
            model_name,
        )
        return 0.0

    if (date.today() - PRICES_LAST_UPDATED).days > 180:
        logger.warning("TOKEN_PRICES sin revisar desde %s.", PRICES_LAST_UPDATED)

    prompt_cost     = (prompt_tokens / 1_000_000) * prices["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * prices["completion"]

    return (prompt_cost + completion_cost) * _usd_to_eur_rate()
