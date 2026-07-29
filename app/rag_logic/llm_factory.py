# app/rag_logic/llm_factory.py
"""
Punto ÚNICO donde se crean los clientes de LLM y de embeddings.

Antes de este módulo, 19 sitios repartidos por 10 ficheros instanciaban
`ChatOpenAI` / `OpenAIEmbeddings` directamente. Cambiar de proveedor obligaba
a editar los 19. Ahora el resto del código pide "dame un modelo" y no sabe
—ni le importa— quién se lo sirve.
"""

import logging
import os

from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Equivalencias OpenAI → Bedrock. IDs verificados en eu-west-1 el 29-jul-2026:
# son inference profiles europeos (prefijo "eu."), no IDs de modelo a secas.
# Con "anthropic.claude-..." sin prefijo, Bedrock responde ValidationException.
BEDROCK_MODEL_MAP = {
    "gpt-4o-mini": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "gpt-4o":      "eu.anthropic.claude-sonnet-4-6",
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _provider() -> str:
    """
    Proveedor activo: 'openai' (por defecto) o 'bedrock'.

    Se lee en CADA llamada, no al importar el módulo: así un test o un script
    pueden cambiar el proveedor sin reimportar nada.

    El defecto 'openai' no es casual — si alguien despliega sin definir la
    variable, el sistema se comporta exactamente como antes de esta migración.

    Se normaliza con tolerancia porque una variable de entorno puede llegar de
    muchos sitios (.env, panel de Railway, consola de Lambda, export en shell) y
    cada uno la trata distinto: ' Bedrock ', "BEDROCK", 'bedrock'.
    Las comillas se quitan por el mismo motivo por el que evaluate_rag.py ya
    saneaba OPENAI_API_KEY: llegan pegadas más a menudo de lo que parece.

    Lo que NO se hace: aceptar alias tipo 'aws' o 'amazon'. El valor válido debe
    poder leerse en este fichero; un valor no reconocido cae a OpenAI y AVISA,
    que es preferible a una lista de sinónimos que nadie mantiene.
    """
    # Normalizar PRIMERO y aplicar el defecto DESPUÉS: si se hace al revés, un
    # valor de solo espacios ('   ') no cae al defecto —es truthy— y acaba
    # resolviendo a cadena vacía.
    valor = (os.environ.get("LLM_PROVIDER") or "").strip()
    valor = valor.strip('"').strip("'").strip().lower()
    return valor or "openai"


def _to_bedrock_id(model_name: str) -> str:
    """
    Traduce un nombre de modelo de OpenAI al ID equivalente en Bedrock.

    Si ya viene un ID de Bedrock (empieza por 'eu.', 'global.' o 'anthropic.'),
    se deja pasar tal cual.

    Un modelo desconocido revienta AQUÍ, con nombre y apellido. La alternativa
    —caer a un modelo por defecto— haría indistinguible "pedí este modelo" de
    "me dieron otro", que es el mismo fallo silencioso de cost_calculator.py.
    """
    if model_name.startswith(("eu.", "global.", "anthropic.")):
        return model_name

    if model_name not in BEDROCK_MODEL_MAP:
        raise ValueError(
            f"No hay equivalente en Bedrock para el modelo '{model_name}'. "
            f"Conocidos: {sorted(BEDROCK_MODEL_MAP)}. "
            f"Añádelo a BEDROCK_MODEL_MAP en llm_factory.py."
        )
    return BEDROCK_MODEL_MAP[model_name]


def get_llm(model_name: str, temperature: float = 0.0, **kwargs):
    """
    Devuelve un cliente de chat listo para usar.

    No devuelve el NOMBRE de un modelo: devuelve un objeto con métodos
    (`.invoke()`, `.bind_tools()`, ...). Quien llama no sabe de qué proveedor es.

    `**kwargs` reenvía lo que cada sitio necesite (`callbacks`,
    `callback_manager`, ...). Sin esto, esos argumentos se perderían en
    silencio y dejaría de verse el logging por consola.
    """
    proveedor = _provider()

    if proveedor == "bedrock":
        return ChatBedrockConverse(
            model_id=_to_bedrock_id(model_name),
            temperature=temperature,
            region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-west-1"),
            **kwargs,
        )

    if proveedor != "openai":
        logger.warning(
            "LLM_PROVIDER='%s' no reconocido; usando OpenAI.", proveedor
        )

    return ChatOpenAI(model_name=model_name, temperature=temperature, **kwargs)


def get_embeddings(model: str = None, **kwargs):
    """
    Devuelve el cliente de embeddings. SIEMPRE OpenAI, ignore LLM_PROVIDER.

    Decisión consciente: los vectores guardados en Chroma se generaron con
    `text-embedding-3-small`. Cambiar el modelo de embeddings invalida el
    índice entero (obliga a reingerir el corpus) y además invalidaría las
    métricas de evaluación ya medidas. La generación se migra; el espacio
    vectorial no se toca.
    """
    return OpenAIEmbeddings(model=model or DEFAULT_EMBEDDING_MODEL, **kwargs)
