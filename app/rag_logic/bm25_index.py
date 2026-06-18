# app/rag_logic/bm25_index.py
"""
Índice BM25 persistente.

Antes, BM25 se reconstruía desde cero (full scan de Chroma + re-tokenizado)
en cada cache-miss de la chain (qa_chain.chain_cache), lo cual ocurre con
frecuencia porque la cache-key incluye el path_filter detectado por pregunta.
Aquí lo construimos UNA VEZ durante la ingesta y lo persistimos a disco;
qa_chain.py solo lo deserializa.
"""

import os
import pickle
from typing import List, Optional

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

_BM25_FILENAME = "_bm25_index.pkl"


def _bm25_path(vector_store_path: str) -> str:
    return os.path.join(vector_store_path, _BM25_FILENAME)


def build_bm25_retriever(docs_text: List[str], docs_meta: List[dict]) -> Optional[BM25Retriever]:
    """Construye un BM25Retriever a partir de listas de texto/metadata ya obtenidas."""
    if not docs_text or len(docs_text) != len(docs_meta):
        return None
    docs = [Document(page_content=t, metadata=(m or {})) for t, m in zip(docs_text, docs_meta)]
    return BM25Retriever.from_documents(docs)


def persist_bm25_index(vector_store: Chroma, vector_store_path: str) -> bool:
    """Reconstruye el índice BM25 desde el contenido actual de Chroma y lo persiste a disco."""
    data = vector_store.get(include=["documents", "metadatas"])
    retriever = build_bm25_retriever(data.get("documents", []) or [], data.get("metadatas", []) or [])
    if retriever is None:
        return False
    os.makedirs(vector_store_path, exist_ok=True)
    with open(_bm25_path(vector_store_path), "wb") as f:
        pickle.dump(retriever, f)
    return True


def load_bm25_index(vector_store_path: str) -> Optional[BM25Retriever]:
    """Carga el índice BM25 persistido. None si no existe o no se pudo deserializar."""
    path = _bm25_path(vector_store_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
