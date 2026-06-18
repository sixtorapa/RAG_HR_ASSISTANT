# app/rag_logic/path_utils.py

import re


def norm_path(s: str) -> str:
    """
    Normaliza un path/filename para comparación y filtrado de metadata.
    Misma normalización usada al ingestar (relative_path_norm) y al detectar
    el filtro de documento en una pregunta, para que ambos lados coincidan
    siempre con un match exacto.
    """
    s = (s or "").strip().lower().replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    s = re.sub(r"\s+", " ", s)
    return s
