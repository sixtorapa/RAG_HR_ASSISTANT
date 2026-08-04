# app/rag_logic/llm_factory.py
"""
The SINGLE place where LLM and embedding clients are created.

Before this module, 19 sites across 10 files instantiated `ChatOpenAI` and
`OpenAIEmbeddings` directly, so changing provider meant editing all 19. The rest
of the code now asks for a model and neither knows nor cares who serves it.
"""

import logging
import os

from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

# OpenAI → Bedrock equivalences. These IDs are European inference profiles (the
# "eu." prefix), not bare model IDs: a bare "anthropic.claude-..." makes Bedrock
# answer with ValidationException.
BEDROCK_MODEL_MAP = {
    "gpt-4o-mini": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "gpt-4o":      "eu.anthropic.claude-sonnet-4-6",
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _provider() -> str:
    """
    The active provider: 'openai' (default) or 'bedrock'.

    Read on EVERY call rather than at import time, so a test or a script can
    change provider without re-importing anything.

    The 'openai' default is not incidental: deploying without setting the
    variable behaves exactly as the system did before the migration existed.

    Normalisation is tolerant because an environment variable arrives from many
    places — a .env file, the Railway panel, the Lambda console, a shell export —
    and each treats it differently: ' Bedrock ', "BEDROCK", 'bedrock'. Quotes are
    stripped for the same reason the evaluation script already sanitised
    OPENAI_API_KEY: they come attached more often than you would think.

    What is NOT done: accepting aliases like 'aws' or 'amazon'. A valid value must
    be readable in this file; an unrecognised one falls back to OpenAI and WARNS,
    which beats a list of synonyms nobody maintains.
    """
    # Normalise FIRST and apply the default AFTER: the other way round, a
    # whitespace-only value ('   ') is truthy, misses the default and resolves to
    # an empty string.
    value = (os.environ.get("LLM_PROVIDER") or "").strip()
    value = value.strip('"').strip("'").strip().lower()
    return value or "openai"


def _to_bedrock_id(model_name: str) -> str:
    """
    Translate an OpenAI model name into its Bedrock equivalent.

    A value that is already a Bedrock ID ('eu.', 'global.' or 'anthropic.') passes
    through untouched.

    An unknown model raises HERE, by name. Falling back to a default would make
    "I asked for this model" indistinguishable from "I was given another" — the
    same silent failure mode cost_calculator.py was fixed for.
    """
    if model_name.startswith(("eu.", "global.", "anthropic.")):
        return model_name

    if model_name not in BEDROCK_MODEL_MAP:
        raise ValueError(
            f"No Bedrock equivalent for model '{model_name}'. "
            f"Known: {sorted(BEDROCK_MODEL_MAP)}. "
            f"Add it to BEDROCK_MODEL_MAP in llm_factory.py."
        )
    return BEDROCK_MODEL_MAP[model_name]


def get_llm(model_name: str, temperature: float = 0.0, **kwargs):
    """
    Return a chat client ready to use.

    Not the NAME of a model: an object with methods (`.invoke()`,
    `.bind_tools()`, ...). The caller does not know which provider served it.

    `**kwargs` forwards whatever each site needs (`callbacks`,
    `callback_manager`, ...). Without it those arguments vanish silently and
    console logging stops appearing.
    """
    provider = _provider()

    if provider == "bedrock":
        return ChatBedrockConverse(
            model_id=_to_bedrock_id(model_name),
            temperature=temperature,
            region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-west-1"),
            **kwargs,
        )

    if provider != "openai":
        logger.warning(
            "LLM_PROVIDER='%s' not recognised; using OpenAI.", provider
        )

    return ChatOpenAI(model_name=model_name, temperature=temperature, **kwargs)


def get_embeddings(model: str = None, **kwargs):
    """
    Return the embeddings client. ALWAYS OpenAI, ignoring LLM_PROVIDER.

    A deliberate decision: the vectors in Chroma were built with
    `text-embedding-3-small`. Changing the embedding model invalidates the whole
    index — forcing a full re-ingest — and invalidates the evaluation numbers
    already measured. Generation migrates; the vector space does not.
    """
    return OpenAIEmbeddings(model=model or DEFAULT_EMBEDDING_MODEL, **kwargs)
