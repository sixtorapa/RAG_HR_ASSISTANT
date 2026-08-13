# app/rag_logic/path_utils.py

import re


def norm_path(s: str) -> str:
    """
    Normalise a path or filename for comparison and metadata filtering.
    The same normalisation used at ingest time (relative_path_norm) and when
    el filtro de documento en una pregunta, para que ambos lados coincidan
    siempre con un match exacto.
    """
    s = (s or "").strip().lower().replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    s = re.sub(r"\s+", " ", s)
    return s
