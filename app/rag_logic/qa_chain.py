# app/rag_logic/qa_chain.py

import os
import re
from difflib import SequenceMatcher
from typing import List, Optional, Dict, Tuple, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document, BaseRetriever

# Hybrid search
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

# Rerank (opcional)
try:
    from langchain.retrievers.document_compressors import FlashrankRerank
    _FLASHRANK_AVAILABLE = True
except Exception:
    FlashrankRerank = None
    _FLASHRANK_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate


# ==================== UTILIDADES ====================

def _norm(s: str) -> str:
    # Normaliza: lower, trim, espacios y separadores de path (Windows -> POSIX)
    s = (s or "").strip().lower().replace("\\", "/")
    s = re.sub(r"/+", "/", s)              # colapsa //// -> /
    s = re.sub(r"\s+", " ", s)             # colapsa espacios
    return s


def _stem(filename: str) -> str:
    base = os.path.basename(filename or "")
    base = re.sub(r"\.(pdf|pptx|ppt)$", "", base, flags=re.IGNORECASE)
    return _norm(base)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ==================== CLASE FILTRADO (PYTHON) ====================

class SmartPathRetriever(BaseRetriever):
    """
    Retriever wrapper que filtra resultados en Python.

    Filtro DURO:
    - Si el filtro parece ruta/archivo (contiene '/' o termina en .pdf/.pptx/.ppt) => match EXACTO
      contra relative_path / filename / source_file (normalizados).
    - Si el filtro parece un "stem" (ej. honimunn) => match EXACTO por stem del filename.
    """
    vector_retriever: BaseRetriever
    path_filter: str
    max_docs: int = 18

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs = self.vector_retriever.get_relevant_documents(query)

        filtro_raw = (self.path_filter or "").strip()
        filtro = _norm(filtro_raw)

        # Heurística: si parece path/archivo => exact
        looks_like_path = ("/" in filtro) or bool(re.search(r"\.(pdf|pptx|ppt)$", filtro, flags=re.IGNORECASE))

        filtered: List[Document] = []
        for d in docs:
            meta = d.metadata or {}

            rel = _norm(meta.get("relative_path", ""))
            fname = _norm(meta.get("filename", ""))
            src = _norm(meta.get("source_file", meta.get("source", "")))

            if looks_like_path:
                # 🔒 DURO: solo exact match (evita contaminación)
                hay_match = (filtro == rel) or (filtro == fname) or (filtro == src)
            else:
                # 🔒 DURO: match exacto por stem
                hay_match = (filtro == _stem(fname)) or (filtro == _stem(src)) or (filtro == _stem(rel))

            if hay_match:
                filtered.append(d)

        print(f"🔎 SmartPathRetriever(HARD): {len(docs)} -> {len(filtered)} docs | filtro='{self.path_filter}'")
        return filtered[: self.max_docs]


# ==================== 2-PASS RETRIEVAL (DOC SHORTLIST) ====================

def _doc_id_from_meta(meta: dict) -> str:
    """
    Identificador estable de "documento" para agrupar chunks.
    Priorizamos relative_path (si existe), si no filename/source_file/source.
    """
    meta = meta or {}
    rel = (meta.get("relative_path") or "").strip()
    fname = (meta.get("filename") or "").strip()
    src = (meta.get("source_file") or meta.get("source") or "").strip()
    return rel or fname or src or "unknown"


def _tokenize_query_for_boost(query: str) -> List[str]:
    """
    Tokens simples para boost de doc (evita stopwords básicas y tokens cortos).
    """
    q = _norm(query)
    toks = re.findall(r"[a-z0-9áéíóúüñ]+", q, flags=re.IGNORECASE)
    stop = {"de", "la", "el", "y", "o", "para", "por", "con", "del", "los", "las", "un", "una", "que", "en"}
    out: List[str] = []
    for t in toks:
        if len(t) < 4:
            continue
        if t in stop:
            continue
        out.append(t)

    # dedupe conservando orden
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)

    return uniq[:8]


def _pick_top_docs_from_candidates(
    query: str,
    candidates: List[Document],
    top_n: int = 2,
    min_votes: int = 2,
) -> List[str]:
    """
    Elige top documentos por "votos" (cuántos chunks aparecen en candidatos)
    + boost si el doc parece contener tokens importantes del query.
    Devuelve lista de doc_ids (relative_path o filename/source).
    """
    if not candidates:
        return []

    tokens = _tokenize_query_for_boost(query)

    stats: Dict[str, dict] = {}
    for d in candidates:
        meta = d.metadata or {}
        doc_id = _doc_id_from_meta(meta)
        if doc_id not in stats:
            stats[doc_id] = {"votes": 0, "boost": 0.0, "meta": meta}
        stats[doc_id]["votes"] += 1

        # Boost simple: si tokens aparecen en filename/relative_path/folder_context
        hay = " ".join([
            _norm(meta.get("filename", "")),
            _norm(meta.get("relative_path", "")),
            _norm(meta.get("folder_context", "")),
        ])
        for t in tokens:
            if t and t in hay:
                stats[doc_id]["boost"] += 0.35

    ranked = sorted(
        stats.items(),
        key=lambda kv: (kv[1]["votes"] + kv[1]["boost"], kv[1]["votes"]),
        reverse=True,
    )

    # Si ni el top-1 alcanza min_votes, consideramos "no hay claridad" -> fallback
    top_doc_id, top_info = ranked[0]
    if top_info["votes"] < min_votes:
        return []

    winners: List[str] = []
    for doc_id, info in ranked:
        if info["votes"] < min_votes and len(winners) > 0:
            break
        winners.append(doc_id)
        if len(winners) >= top_n:
            break

    return winners



class MultiPathRetriever(BaseRetriever):
    """
    Igual que SmartPathRetriever, pero permite varios path_filters (OR).
    Filtra DURO por exact match (path/archivo) o por stem (si no parece path).
    """
    vector_retriever: BaseRetriever
    path_filters: List[str]
    max_docs: int = 22

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs = self.vector_retriever.get_relevant_documents(query)

        filtros_raw = [f for f in (self.path_filters or []) if (f or "").strip()]
        filtros = [_norm(f) for f in filtros_raw]

        # separa filtros "parecen path/archivo" vs "stems"
        path_like: List[str] = []
        stem_like: List[str] = []
        for f_raw, f_norm in zip(filtros_raw, filtros):
            looks_like_path = ("/" in f_norm) or bool(re.search(r"\.(pdf|pptx|ppt)$", f_norm, flags=re.IGNORECASE))
            if looks_like_path:
                path_like.append(f_norm)
            else:
                stem_like.append(f_norm)

        filtered: List[Document] = []
        for d in docs:
            meta = d.metadata or {}
            rel = _norm(meta.get("relative_path", ""))
            fname = _norm(meta.get("filename", ""))
            src = _norm(meta.get("source_file", meta.get("source", "")))

            ok = False

            # exact path/filename/source
            if path_like:
                ok = (rel in path_like) or (fname in path_like) or (src in path_like)

            # stem exact (si no pasó por path_like)
            if (not ok) and stem_like:
                ok = (_stem(fname) in stem_like) or (_stem(src) in stem_like) or (_stem(rel) in stem_like)

            if ok:
                filtered.append(d)

        print(f"🔎 MultiPathRetriever(HARD): {len(docs)} -> {len(filtered)} docs | filtros={filtros_raw}")
        return filtered[: self.max_docs]


class TwoPassDocShortlistRetriever(BaseRetriever):
    """
    1) Primer pase: base_retriever global (ensemble actual).
    2) Elegir top docs.
    3) Segundo pase: volver a recuperar con filtro duro multi-doc.
    """
    base_retriever: BaseRetriever
    top_docs: int = 2
    min_votes: int = 2
    first_pass_k: int = 14
    max_docs: int = 22

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        candidates = self.base_retriever.get_relevant_documents(query) or []
        candidates = candidates[: max(self.first_pass_k, 1)]

        winners = _pick_top_docs_from_candidates(
            query=query,
            candidates=candidates,
            top_n=self.top_docs,
            min_votes=self.min_votes,
        )

        if not winners:
            return candidates[: self.max_docs]

        print(f"🎯 TwoPass: docs ganadores -> {winners}")

        filtered_retriever = MultiPathRetriever(
            vector_retriever=self.base_retriever,
            path_filters=winners,
            max_docs=self.max_docs,
        )
        return filtered_retriever.get_relevant_documents(query)


# ==================== PROMPTS ====================

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Rewrite the user's question as a standalone, self-contained question. "
         "If the question references a document (PDF/XLSX/PPTX), preserve the document name in the final question."),
        ("human", "Chat history:\n{chat_history}\n\nQuestion:\n{question}\n\nStandalone question:"),
    ]
)

SISTEMA_ROL = """You are a PROFESSIONAL HR KNOWLEDGE BASE ASSISTANT.

Your objective:
- Answer EXCLUSIVELY based on the CONTEXT retrieved from internal documents (PDFs, XLSX, PPTX).
- If the user asks about a specific document ("summarise document X" or "according to PDF X"), focus on THAT document.
- Always cite sources at the end in the format: (source: filename.pdf, p. N) or (source: filename.xlsx, sheet Name)

Rules:
- If context is available, ALWAYS build a useful answer from it (even if partial).
- Only say "not found" when the context is empty or completely irrelevant.
- Use available metadata: filename, relative_path, page/slide numbers where present.
"""

INSTRUCCION_FORMATO = """
Format:
- Clear and direct answer.
- Use bullet points where appropriate.
- Close with a "Sources:" block listing files and page/slide numbers.
"""

def construir_template_qa(instruccion_personalizada: str = "") -> PromptTemplate:
    sistema = instruccion_personalizada.strip() if (instruccion_personalizada or "").strip() else SISTEMA_ROL
    template = f"""{sistema}

{INSTRUCCION_FORMATO}

---DOCUMENT CONTEXT---
{{context}}

---CHAT HISTORY---
{{chat_history}}

---USER QUESTION---
{{question}}

---YOUR ANSWER---
"""
    return PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])


# ==================== CATÁLOGO DE DOCUMENTOS ====================

_catalog_cache: Dict[str, Dict[str, str]] = {}
# cache_key -> { "filename_lower": "relative_path_lower" }
# además se usa para matching por stem

def _build_doc_catalog(vector_store: Chroma, cache_key: str) -> Dict[str, str]:
    if cache_key in _catalog_cache:
        return _catalog_cache[cache_key]

    data = vector_store.get(include=["metadatas"])
    out: Dict[str, str] = {}
    for m in data.get("metadatas", []) or []:
        if not m:
            continue
        fname = m.get("filename") or ""
        rel = m.get("relative_path") or fname
        if fname:
            out[_norm(fname)] = _norm(rel)
            out[_stem(fname)] = _norm(rel)  # clave adicional por stem

    _catalog_cache[cache_key] = out
    print(f"📚 Catálogo de docs: {len(out)} claves (cache_key={cache_key})")
    return out


def _detect_doc_filter(question: str, catalog: Dict[str, str]) -> Optional[str]:
    """
    Devuelve un string para filtrar (relative_path o filename) si detecta que el usuario
    está refiriéndose a un documento concreto.
    """
    q = _norm(question)

    # 1) Si viene con extensión explícita
    m = re.search(r"([a-z0-9_\-\. ]+)\.(pdf|pptx|ppt)\b", q, flags=re.IGNORECASE)
    if m:
        candidate = _norm(m.group(0))
        # match directo
        if candidate in catalog:
            return catalog[candidate]
        # match por stem
        st = _stem(candidate)
        if st in catalog:
            return catalog[st]
        return candidate  # al menos intentarlo

    # 2) Match fuzzy contra stems conocidos (solo si la pregunta parece pedir doc)
    looks_doc = any(k in q for k in ["pdf", "ppt", "pptx", "documento", "presentación", "informe", "según", "del archivo"])
    if not looks_doc:
        return None

    # 2.A) Match directo por palabra completa
    for k in catalog.keys():
        if "." in k:
            continue
        if re.search(rf"\b{re.escape(k)}\b", q):
            return catalog[k]

    # 2.B) Fuzzy fallback
    best_key = None
    best_score = 0.0
    for k in catalog.keys():
        if "." in k:
            continue
        score = _ratio(q, k)
        if score > best_score:
            best_score = score
            best_key = k

    if best_key and best_score >= 0.62:
        return catalog[best_key]

    return None


# ==================== CACHÉ DE CADENAS ====================

chain_cache: Dict[str, ConversationalRetrievalChain] = {}


# ==================== FUNCIÓN PRINCIPAL ====================

def get_conversational_qa_chain(
    project_id: str,
    vector_store_path: str,
    model_name: str,
    project_settings: Optional[dict] = None,
    search_kwargs_override: Optional[dict] = None,
):
    """
    Crea una ConversationalRetrievalChain optimizada para:
    - 50 PDFs/PPTs
    - alta precisión
    - “doc-aware retrieval” (si se menciona un doc, filtra)
    """

    if project_settings is None:
        project_settings = {}

    # --- LLM ---
    temperature = float(project_settings.get("temperature", 0.0))
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # --- Embeddings / Vector Store ---
    embedding_model = os.environ.get("UP_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embeddings,
    )

    # --- Catalog (para detectar doc por nombre) ---
    catalog_key = f"{project_id}::{vector_store_path}"
    catalog = _build_doc_catalog(vector_store, cache_key=catalog_key)

    # --- Override manual (si viene desde herramienta) ---
    forced_filter = None
    if search_kwargs_override and "python_path_filter" in search_kwargs_override:
        forced_filter = search_kwargs_override["python_path_filter"]

    # Auto-detección: si el usuario menciona doc, filtramos
    auto_filter = None
    if not forced_filter:
        auto_filter = _detect_doc_filter(project_settings.get("last_user_question", ""), catalog)

    path_filter = forced_filter or auto_filter

    # --- Cache key (incluye filtro si existe) ---
    cache_key = f"{project_id}::{model_name}::{path_filter or 'NO_FILTER'}"
    if cache_key in chain_cache:
        return chain_cache[cache_key]

    # ==================== Retriever base (MMR + k alto) ====================
    k_base = int(project_settings.get("k_base", 28 if not path_filter else 60))
    fetch_k = max(k_base * 4, 80)

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k_base, "fetch_k": fetch_k, "lambda_mult": 0.55},
    )

    # ==================== BM25 (opcional) ====================
    ensemble_retriever: BaseRetriever = vector_retriever
    try:
        data = vector_store.get(include=["documents", "metadatas"])
        docs_text = data.get("documents", []) or []
        docs_meta = data.get("metadatas", []) or []
        if 0 < len(docs_text) == len(docs_meta):
            docs_bm25 = [Document(page_content=t, metadata=(m or {})) for t, m in zip(docs_text, docs_meta)]
            bm25 = BM25Retriever.from_documents(docs_bm25)
            bm25.k = min(30, max(10, k_base))
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25],
                weights=[0.55, 0.45],
            )
            print(f"✅ BM25 activo: {len(docs_bm25)} chunks indexados para BM25")
    except Exception as e:
        print(f"⚠️ BM25 desactivado por error: {e}")
        ensemble_retriever = vector_retriever

    # ==================== Filtro por documento (si aplica) ====================
    final_retriever: BaseRetriever = ensemble_retriever

    if path_filter:
        # Si el usuario menciona doc/ruta, filtro duro a 1 doc
        final_retriever = SmartPathRetriever(
            vector_retriever=ensemble_retriever,
            path_filter=path_filter,
            max_docs=22,
        )
    else:
        # ✅ Two-pass retrieval (doc shortlist) para evitar mezclar documentos en preguntas genéricas
        two_pass_enabled = bool(project_settings.get("two_pass_enabled", True))
        if two_pass_enabled:
            final_retriever = TwoPassDocShortlistRetriever(
                base_retriever=ensemble_retriever,
                top_docs=int(project_settings.get("two_pass_top_docs", 2)),
                min_votes=int(project_settings.get("two_pass_min_votes", 2)),
                first_pass_k=int(project_settings.get("two_pass_first_pass_k", 14)),
                max_docs=int(project_settings.get("two_pass_max_docs", 22)),
            )

    # ==================== Rerank / Compression (si está disponible) ====================
    if _FLASHRANK_AVAILABLE:
        try:
            top_n = int(project_settings.get("rerank_top_n", 14 if not path_filter else 18))
            compressor = FlashrankRerank(top_n=top_n)
            final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=final_retriever,
            )
            print(f"✅ Flashrank activo (top_n={top_n})")
        except Exception as e:
            print(f"⚠️ Flashrank no disponible: {e}")

    # ==================== Chain ====================
    QA_PROMPT = construir_template_qa(project_settings.get("system_instruction", ""))

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=final_retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=True,
    )

    chain_cache[cache_key] = chain
    return chain
