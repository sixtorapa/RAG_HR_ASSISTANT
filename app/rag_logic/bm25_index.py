# app/rag_logic/bm25_index.py
"""
Persistent BM25 index.

BM25 needs corpus-wide statistics, so building it means a full Chroma scan plus
re-tokenising everything. That cost belongs to ingestion, not to a query: the
chain cache key includes the per-question path filter, so cache misses are
frequent and rebuilding on each one would charge every user for the ingest.

Built ONCE during ingestion and persisted to disk; qa_chain.py only loads it.
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
    """Build a BM25Retriever from already-fetched text and metadata lists."""
    if not docs_text or len(docs_text) != len(docs_meta):
        return None
    docs = [Document(page_content=t, metadata=(m or {})) for t, m in zip(docs_text, docs_meta)]
    return BM25Retriever.from_documents(docs)


def persist_bm25_index(vector_store: Chroma, vector_store_path: str) -> bool:
    """Rebuild the BM25 index from the current Chroma contents and persist it to disk."""
    data = vector_store.get(include=["documents", "metadatas"])
    retriever = build_bm25_retriever(data.get("documents", []) or [], data.get("metadatas", []) or [])
    if retriever is None:
        return False
    os.makedirs(vector_store_path, exist_ok=True)
    with open(_bm25_path(vector_store_path), "wb") as f:
        pickle.dump(retriever, f)
    return True


def load_bm25_index(vector_store_path: str) -> Optional[BM25Retriever]:
    """Load the persisted BM25 index. None if absent or if it could not be deserialised."""
    path = _bm25_path(vector_store_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
