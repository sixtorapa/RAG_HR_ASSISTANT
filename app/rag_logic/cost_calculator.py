# app/rag_logic/cost_calculator.py

from flask import current_app

# Precios en USD por 1 millón de tokens (a fecha 2025 - ¡revisar y actualizar!)
# Fuente: https://openai.com/es-ES/api/pricing/

TOKEN_PRICES = {
    "gpt-4o-mini": {
        "prompt": 0.15,
        "completion": 0.60
    },
    "gpt-4o": {
        "prompt": 2.50,
        "completion": 10.00
    }
}



def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calcula el costo de una llamada a la API de OpenAI basándose en el modelo y el uso de tokens.
    """
    if model_name not in TOKEN_PRICES:
        # Si el modelo no está en nuestra lista, no podemos calcular el costo.
        # Podríamos lanzar un error o simplemente devolver 0.
        print(f"Advertencia: Modelo '{model_name}' no encontrado en la lista de precios. No se calculará el costo.")
        return 0.0

    prices = TOKEN_PRICES[model_name]
    
    # Los precios están por 1 millón de tokens, así que dividimos el conteo por 1,000,000
    prompt_cost = (prompt_tokens / 1_000_000) * prices["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * prices["completion"]
    
    total_cost_usd = prompt_cost + completion_cost
    conversion_rate = current_app.config.get("USD_TO_EUR_RATE", 1.0) # 1.0 como fallback
    total_cost_eur = total_cost_usd * conversion_rate
    
    return total_cost_eur