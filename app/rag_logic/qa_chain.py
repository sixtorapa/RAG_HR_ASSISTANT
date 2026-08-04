# app/rag_logic/qa_chain.py

import os
import re
from difflib import SequenceMatcher
from typing import List, Optional, Dict, Tuple, Any, Callable

from .llm_factory import get_llm, get_embeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document, BaseRetriever

# Hybrid search
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from .path_utils import norm_path
from .bm25_index import build_bm25_retriever, load_bm25_index

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
    return norm_path(s)


def _stem(filename: str) -> str:
    base = os.path.basename(filename or "")
    base = re.sub(r"\.(pdf|pptx|ppt)$", "", base, flags=re.IGNORECASE)
    return _norm(base)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ==================== PREFILTRO NATIVO POR METADATA ====================

def _combine_filters(*filters: Optional[dict]) -> Optional[dict]:
    """Combina filtros de Chroma con $and. None se ignora."""
    present = [f for f in filters if f]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return {"$and": present}


# ==================== FILTRO DE GRANULARIDAD (macro vs micro) ====================
# La ingesta indexa DOS tamaños del mismo text en la misma colección: macro
# (páginas agrupadas, ~350 palabras) y micro (250 tokens, con parent_chunk_id).
# Medido el 3-ago-2026 sobre el índice real: **153 de los 154 micro son subcadena
# literal de su macro**. Es decir, el mismo contenido compite consigo mismo en
# cada búsqueda, y cuando gana dos veces, el LLM recibe el mismo text duplicado.
# En cuatro preguntas del golden set, el 21% de los caracteres recuperados eran
# repeticiones — y hasta el 56% en las que caen sobre un solo documento.
#
# El diseño parent-child dice "busca por el hijo, que es preciso, y cita por el
# padre, que tiene el contexto". Buscar por los dos es justo lo que ese diseño
# no dice. Este filtro aplica la mitad que sí se puede aplicar today.
#
# Se controla por variable de entorno para poder medirlo y revertirlo sin tocar
# código, igual que FLASHRANK_ENABLED:
#   RETRIEVAL_CHUNK_TYPE=micro  (por defecto) — solo children
#   RETRIEVAL_CHUNK_TYPE=macro                — solo padres
#   RETRIEVAL_CHUNK_TYPE=all                  — comportamiento anterior
def _chunk_type_filter() -> Optional[dict]:
    value = (os.environ.get("RETRIEVAL_CHUNK_TYPE") or "micro").strip().lower()
    if value in ("all", "any", ""):
        return None
    if value not in ("micro", "macro"):
        print(f"⚠️ RETRIEVAL_CHUNK_TYPE='{value}' no reconocido; usando 'micro'.")
        value = "micro"
    return {"chunk_type": value}


def _build_scoped_retriever(
    vector_store: Chroma,
    values: List[str],
    k_base: int,
    fetch_k: int,
    max_docs: int,
    metadata_field: str = "relative_path_norm",
    security_filter: Optional[dict] = None,
) -> BaseRetriever:
    """
    Prefiltro NATIVO por metadata, en vez de lanzar la búsqueda sobre todo el corpus
    y descartar results en Python después (lo que hacían SmartPathRetriever/
    MultiPathRetriever). Reutilizable para distintos campos de metadata: por archivo
    concreto (`relative_path_norm`) o por departamento (`department`).

    `security_filter` (si se pasa) es un filtro de Chroma que SIEMPRE se combina con
    AND sobre el filtro funcional — es el guardarril de control de acceso por
    departamento (ver get_conversational_qa_chain). Nunca se ignora ni se aplica solo
    "si hay tiempo": un documento fuera del security_filter no puede salir nunca,
    aunque el filtro funcional (archivo concreto, departamento detectado) lo pidiera.

    - Vector leg: Chroma recibe `filter` en el propio search_kwargs -> el MMR search
      solo considera los chunks que matchean, vía el índice de metadata de Chroma.
    - BM25 leg: se construye SOLO con los chunks de ese filtro (vector_store.get(where=...)),
      no con el corpus completo -> rápido independientemente de cuántos documentos haya en total.
    """
    norm_values = [norm_path(v) for v in (values or []) if (v or "").strip()]
    functional_filter = (
        ({metadata_field: norm_values[0]} if len(norm_values) == 1 else {metadata_field: {"$in": norm_values}})
        if norm_values else None
    )
    chroma_filter = _combine_filters(functional_filter, security_filter, _chunk_type_filter())

    if not chroma_filter:
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": max_docs, "fetch_k": fetch_k, "lambda_mult": 0.55},
        )

    scoped_vector = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max_docs, "fetch_k": fetch_k, "lambda_mult": 0.55, "filter": chroma_filter},
    )

    scoped_retriever: BaseRetriever = scoped_vector
    try:
        data = vector_store.get(where=chroma_filter, include=["documents", "metadatas"])
        docs_text = data.get("documents", []) or []
        bm25 = build_bm25_retriever(docs_text, data.get("metadatas", []) or [])
        if bm25 is not None:
            bm25.k = min(max_docs, max(10, len(docs_text)))
            scoped_retriever = EnsembleRetriever(retrievers=[scoped_vector, bm25], weights=[0.55, 0.45])
    except Exception as e:
        print(f"⚠️ BM25 (scoped) desactivado por error: {e}")

    print(f"🔎 ScopedRetriever(NATIVE filter): {chroma_filter}")
    return scoped_retriever


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



class TwoPassDocShortlistRetriever(BaseRetriever):
    """
    1) Primer pase: base_retriever global (ensemble sin filtrar, para descubrir qué doc(s)
       son relevantes cuando el usuario no menciona uno explícitamente).
    2) Elegir top docs por votos/boost.
    3) Segundo pase: prefiltro NATIVO multi-doc vía scoped_retriever_factory
       (_build_scoped_retriever), no post-filtrado en Python sobre una segunda
       búsqueda sin filtrar.
    """
    base_retriever: BaseRetriever
    scoped_retriever_factory: Callable[[List[str], int], BaseRetriever]
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

        scoped_retriever = self.scoped_retriever_factory(winners, self.max_docs)
        return scoped_retriever.get_relevant_documents(query)


# ==================== EXPANSIÓN HIJO → PADRE ====================

class ParentExpansionRetriever(BaseRetriever):
    """
    Busca por el hijo y devuelve el padre.

    Es la mitad que faltaba de la arquitectura parent-child. La ingesta ya
    guardaba `parent_chunk_id` en cada micro-chunk, pero nadie lo usaba: la
    búsqueda iba contra los dos tamaños a la vez y el mismo text competía
    consigo mismo (153 de 154 micro son subcadena literal de su macro).

    La tensión del chunking es conocida: el trozo pequeño se RECUPERA mejor
    —su embedding es específico y tiene menos ruido— pero RESPONDE peor,
    porque le falta context. En vez de buscar el tamaño mágico, se usan los
    dos para lo que cada uno hace bien: se busca con el hijo y se entrega el
    padre.

    El efecto sobre la precisión es directo: varios children del mismo padre
    colapsan en UNA sola entrada, así que desaparece la redundancia que
    medimos (21% de los caracteres recuperados eran text repetido, hasta un
    56% en preguntas sobre un solo documento).

    El orden se conserva: la posición del padre la fija su primer hijo, así
    que el ranking del retriever de abajo no se pierde.

    Se desactiva con PARENT_EXPANSION=0.
    """

    base_retriever: BaseRetriever
    vector_store: Chroma
    max_docs: int = 12

    def _fetch_parent(self, parent_id: str) -> Optional[Document]:
        try:
            data = self.vector_store.get(
                where={"chunk_id": parent_id},
                include=["documents", "metadatas"],
                limit=1,
            )
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            if texts:
                return Document(page_content=texts[0], metadata=(metas[0] if metas else {}) or {})
        except Exception as e:
            print(f"⚠️ Expansión al padre falló para {parent_id}: {e}")
        return None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        children = self.base_retriever.get_relevant_documents(query) or []

        seen: set = set()
        out: List[Document] = []

        for d in children:
            meta = d.metadata or {}
            parent_id = (meta.get("parent_chunk_id") or "").strip()

            # Sin padre (ya es macro, o viene de un formato sin jerarquía):
            # se queda tal cual, pero igualmente deduplicado.
            key = parent_id or (meta.get("chunk_id") or d.page_content[:120])
            if key in seen:
                continue
            seen.add(key)

            out.append(self._fetch_parent(parent_id) or d if parent_id else d)
            if len(out) >= self.max_docs:
                break

        if out:
            print(f"👪 Expansión hijo→padre: {len(children)} chunks → {len(out)} documentos únicos")
        return out


def _parent_expansion_enabled() -> bool:
    return str(os.environ.get("PARENT_EXPANSION", "1")).strip().lower() in ("1", "true", "yes")


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
            out[_stem(fname)] = _norm(rel)  # key adicional por stem

    _catalog_cache[cache_key] = out
    print(f"📚 Catálogo de docs: {len(out)} keys (cache_key={cache_key})")
    return out


def _detect_doc_filter(question: str, catalog: Dict[str, str]) -> Optional[str]:
    """
    Devuelve un string para filtrar (relative_path o filename) si detecta que el usuario
    está refiriéndose a un documento concreto.
    """
    q = _norm(question)

    # 1) Si viene con extensión explícita
    # Sin espacio en la clase de caracteres: si lo incluyéramos, capturaría toda la
    # frase anterior al nombre de archivo en vez de solo el nombre (p.ej. "según el
    # documento X.pdf" -> "documento x.pdf" en lugar de "x.pdf").
    m = re.search(r"([a-z0-9áéíóúüñ_\-\.]+)\.(pdf|pptx|ppt)\b", q, flags=re.IGNORECASE)
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


# ==================== CLASIFICADOR DE DEPARTAMENTO ====================
# Heurística por keywords (rápida, gratis, determinista). Si se quisiera más
# precisión a costa de latencia/coste, esto se sustituiría por una llamada LLM
# (clasificación zero-shot contra la lista de departamentos) o por similitud de
# embeddings contra una descripción corta de cada departamento — el resto del
# pipeline (filtro nativo por metadata "department") no cambiaría.

_DEPARTMENT_KEYWORDS: Dict[str, List[str]] = {
    "compensation_benefits": [
        "salary", "salaries", "pay", "compensation", "bonus", "stock option", "esop",
        "benefit", "pension", "payroll", "salario", "sueldo", "compensacion", "bono",
        "pension", "beneficio", "nomina", "banda salarial", "aumento",
    ],
    "recruitment_talent": [
        "hiring", "recruit", "interview", "candidate", "referral", "background check",
        "internship", "offer letter", "contratacion", "reclutamiento", "entrevista",
        "candidato", "referido", "becario", "practicas", "oferta de trabajo",
    ],
    "performance_management": [
        "performance review", "okr", "goal setting", "pip", "improvement plan",
        "promotion", "calibration", "evaluacion de desempeno", "objetivos",
        "ascenso", "promocion", "calibracion", "revision de desempeno",
    ],
    "onboarding_people_ops": [
        "onboarding", "new hire", "probation", "org chart", "reporting line",
        "transfer", "mobility", "vacation", "annual leave", "sick leave",
        "incorporacion", "nuevo empleado", "periodo de prueba", "organigrama",
        "traslado", "movilidad interna", "vacaciones", "dias libres", "baja medica",
    ],
    "learning_development": [
        "training", "course", "certification", "conference", "mentorship", "mentor",
        "learning", "lms", "formacion", "curso", "certificacion", "conferencia",
        "mentoria", "aprendizaje",
    ],
    "health_safety_wellbeing": [
        "health", "safety", "wellness", "ergonomic", "mental health", "therapy",
        "incident", "evacuation", "salud", "seguridad", "bienestar", "ergonomia",
        "salud mental", "terapia", "incidente", "evacuacion",
    ],
    "legal_compliance_conduct": [
        "code of conduct", "harassment", "discrimination", "whistleblower", "gdpr",
        "privacy", "conflict of interest", "disciplinary", "investigation",
        "acoso", "discriminacion", "privacidad", "conflicto de interes",
        "disciplinario", "investigacion", "etica",
    ],
    "it_workplace_policies": [
        "remote work", "hybrid", "it security", "password", "vpn", "equipment",
        "byod", "visitor", "badge", "software license", "trabajo remoto", "hibrido",
        "seguridad informatica", "contrasena", "equipo", "visitante", "licencia",
    ],
    "diversity_equity_inclusion": [
        "dei", "diversity", "inclusion", "erg", "parental leave", "accessibility",
        "accommodation", "pay equity", "diversidad", "inclusion", "permiso parental",
        "accesibilidad", "equidad salarial",
    ],
    "finance_travel_expenses": [
        "travel", "expense", "per diem", "corporate card", "procurement", "vendor",
        "budget", "viaje", "gasto", "dieta", "tarjeta corporativa", "presupuesto",
        "provider", "reembolso",
    ],
}


def _detect_department(question: str) -> Optional[str]:
    """
    Heurística de clasificación de departamento por keywords.
    Devuelve None si no hay una señal clara (evita filtrar mal y perder el
    documento correcto) — en ese caso el caller debe caer al flujo sin filtro.
    """
    q = _norm(question)
    if not q:
        return None

    scores: Dict[str, int] = {}
    for dept, keywords in _DEPARTMENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score:
            scores[dept] = score

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_dept, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Solo confiamos si hay una señal clara y sin ambigüedad entre dos departamentos
    if top_score >= 1 and top_score > second_score:
        return top_dept

    return None


# ==================== GUARDARRIL: CONTROL DE ACCESO POR DEPARTAMENTO ====================
# A diferencia de _detect_department (heurística, optimización, puede fallar sin
# consecuencias graves), esto es un control de seguridad: SIEMPRE se aplica con AND
# sobre cualquier filtro funcional (ver _build_scoped_retriever/_combine_filters),
# nunca se ignora ni se "mejora" con detección de intención.

def _build_security_filter(allowed_departments: Optional[List[str]]) -> Optional[dict]:
    """
    allowed_departments:
        None -> sin restricción (admin / User.get_allowed_departments()).
        []   -> sin acceso a ningún departamento (fail closed: el value por defecto
                 si un caller olvida pasar este parámetro, ver get_conversational_qa_chain).
        list -> restringido exactamente a esos departamentos.
    """
    if allowed_departments is None:
        return None
    norm_allowed = sorted({norm_path(d) for d in allowed_departments if (d or "").strip()})
    if not norm_allowed:
        # Valor de department que ningún chunk real puede tener -> 0 results,
        # en vez de depender de cómo Chroma trate un $in vacío.
        return {"department": "__no_access__"}
    return {"department": {"$in": norm_allowed}}


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

    # --- Guardarril: control de acceso por departamento (RBAC) ---
    # Fail closed por defecto: si el caller no pasa "allowed_departments", se asume
    # sin acceso a ningún departamento, no acceso total. Lo inyecta routes.py desde
    # current_user.get_allowed_departments() (None para admin = sin restricción).
    allowed_departments = project_settings.get("allowed_departments", [])
    security_filter = _build_security_filter(allowed_departments)

    # --- LLM ---
    temperature = float(project_settings.get("temperature", 0.0))
    llm = get_llm(model_name, temperature)

    # --- Embeddings / Vector Store ---
    embedding_model = os.environ.get("UP_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = get_embeddings(embedding_model)
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embeddings,
    )

    # --- Catalog (para detectar doc por name) ---
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

    # Detección de departamento — solo relevante si no hay path_filter explícito.
    # Se calcula aquí (no más abajo) para que entre en la cache key: dos preguntas
    # sin path_filter pero de departamentos distintos NO deben compartir chain/retriever.
    detected_department = None if path_filter else _detect_department(project_settings.get("last_user_question", ""))

    # --- Cache key (incluye filtro/departamento Y el alcance de seguridad) ---
    # Crítico: dos usuarios con distinto allowed_departments NUNCA deben compartir
    # chain/retriever cacheados, aunque hagan la misma question.
    security_scope_key = "ADMIN" if allowed_departments is None else ",".join(sorted(allowed_departments)) or "NONE"
    # La granularidad entra en la clave por el mismo motivo que la ACL: dos
    # configuraciones distintas no pueden compartir retriever cacheado, o al
    # cambiar la variable seguirías sirviendo la cadena vieja hasta reiniciar.
    granularity = (_chunk_type_filter() or {}).get("chunk_type", "all")
    cache_key = (
        f"{project_id}::{model_name}::{path_filter or ('DEPT:' + detected_department if detected_department else 'NO_FILTER')}"
        f"::ACL:{security_scope_key}::GRANO:{granularity}::PAD:{int(_parent_expansion_enabled())}"
    )
    if cache_key in chain_cache:
        return chain_cache[cache_key]

    # ==================== Retriever base (MMR + k alto) ====================
    k_base = int(project_settings.get("k_base", 28 if not path_filter else 60))
    fetch_k = max(k_base * 4, 80)

    # Closure que el TwoPassDocShortlistRetriever usa en su 2º pase (prefiltro nativo multi-doc).
    # security_filter va SIEMPRE, aunque el primer pase ya estuviera acotado: defensa en
    # profundidad, no cuesta nada extra y evita que un futuro cambio en el primer pase
    # abra un agujero aquí sin que nadie se dé cuenta.
    def _scoped_retriever_factory(values: List[str], max_docs: int) -> BaseRetriever:
        return _build_scoped_retriever(
            vector_store=vector_store,
            values=values,
            k_base=k_base,
            fetch_k=fetch_k,
            max_docs=max_docs,
            metadata_field="relative_path_norm",
            security_filter=security_filter,
        )

    if path_filter:
        # El usuario menciona doc/ruta -> prefiltro NATIVO directo a 1 documento.
        # security_filter va con AND: si ese documento no está en un departamento
        # permitido para este usuario, esto devuelve 0 results (deny), nunca el doc.
        final_retriever: BaseRetriever = _build_scoped_retriever(
            vector_store=vector_store,
            values=[path_filter],
            k_base=k_base,
            fetch_k=fetch_k,
            max_docs=22,
            metadata_field="relative_path_norm",
            security_filter=security_filter,
        )
    else:
        # detected_department ya se calculó arriba (antes de la cache key).
        # Si hay señal clara, prefiltramos nativamente por "department" ANTES de
        # construir el ensemble, en vez de buscar siempre sobre el corpus completo.
        # Con baja confianza, no filtramos (más seguro perder velocidad que perder
        # el documento correcto) — pero el guardarril de acceso (security_filter) se
        # aplica igual, detectemos departamento o no: es quien decide qué se puede
        # ver, la heurística solo decide qué se mira primero para ir más rápido.
        # El filtro de granularidad obliga a pasar por el retriever con prefiltro
        # nativo: la pata BM25 persistida se construyó con macro Y micro, así que
        # filtrar solo la pata vectorial dejaría entrar los macro por la otra
        # puerta. _build_scoped_retriever reconstruye BM25 con el subconjunto ya
        # filtrado (son ~150 chunks: milisegundos).
        if detected_department or security_filter or _chunk_type_filter():
            if detected_department:
                print(f"🏷️ Departamento detectado: {detected_department}")
            if security_filter:
                print(f"🔒 Guardarril de acceso activo: {security_filter}")
            ensemble_retriever: BaseRetriever = _build_scoped_retriever(
                vector_store=vector_store,
                values=[detected_department] if detected_department else [],
                k_base=k_base,
                fetch_k=fetch_k,
                max_docs=k_base,
                metadata_field="department",
                security_filter=security_filter,
            )
        else:
            vector_retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_base, "fetch_k": fetch_k, "lambda_mult": 0.55},
            )

            # ==================== BM25 (índice persistido, no se reconstruye en cada query) ====================
            ensemble_retriever = vector_retriever
            try:
                bm25 = load_bm25_index(vector_store_path)
                if bm25 is not None:
                    print("✅ BM25 activo (índice persistido)")
                else:
                    # Fallback defensivo: vector store aún no re-indexado con el nuevo flujo persistente.
                    data = vector_store.get(include=["documents", "metadatas"])
                    bm25 = build_bm25_retriever(data.get("documents", []) or [], data.get("metadatas", []) or [])
                    if bm25 is not None:
                        print("⚠️ BM25 sin índice persistido: construido al vuelo (ejecuta ingest.py para persistirlo)")
                if bm25 is not None:
                    bm25.k = min(30, max(10, k_base))
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[vector_retriever, bm25],
                        weights=[0.55, 0.45],
                    )
            except Exception as e:
                print(f"⚠️ BM25 desactivado por error: {e}")
                ensemble_retriever = vector_retriever

        # ✅ Two-pass retrieval (doc shortlist) para evitar mezclar documentos en preguntas genéricas
        final_retriever = ensemble_retriever
        two_pass_enabled = bool(project_settings.get("two_pass_enabled", True))
        if two_pass_enabled:
            final_retriever = TwoPassDocShortlistRetriever(
                base_retriever=ensemble_retriever,
                scoped_retriever_factory=_scoped_retriever_factory,
                top_docs=int(project_settings.get("two_pass_top_docs", 2)),
                min_votes=int(project_settings.get("two_pass_min_votes", 2)),
                first_pass_k=int(project_settings.get("two_pass_first_pass_k", 14)),
                max_docs=int(project_settings.get("two_pass_max_docs", 22)),
            )

    # ==================== Expansión hijo → padre ====================
    # Va DESPUÉS del two-pass y ANTES del rerank: el two-pass trabaja con los
    # children (que es donde está la señal precisa) y el LLM recibe los padres
    # (que es donde está el contexto completo).
    if _parent_expansion_enabled() and (_chunk_type_filter() or {}).get("chunk_type") == "micro":
        final_retriever = ParentExpansionRetriever(
            base_retriever=final_retriever,
            vector_store=vector_store,
            max_docs=int(project_settings.get("parent_expansion_max_docs", 12)),
        )

    # ==================== Rerank / Compression (opt-in) ====================
    flashrank_enabled = str(os.environ.get("FLASHRANK_ENABLED", "0")).strip().lower() in ("1", "true", "yes")

    if _FLASHRANK_AVAILABLE and flashrank_enabled:
        try:
            top_n = int(project_settings.get("rerank_top_n", 14 if not path_filter else 18))
            compressor = FlashrankRerank(
                top_n=top_n,
                model=os.environ.get("FLASHRANK_MODEL_NAME", "ms-marco-MiniLM-L-12-v2"),
                cache_dir=os.environ.get("FLASHRANK_CACHE_DIR", "/opt/flashrank"),
            )
            final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=final_retriever,
            )
            print(f"✅ Flashrank activo (top_n={top_n})")
        except Exception as e:
            print(f"⚠️ Flashrank falló, continúo sin rerank: {e}")
    else:
        if _FLASHRANK_AVAILABLE and not flashrank_enabled:
            print("ℹ️ Flashrank instalado pero desactivado (FLASHRANK_ENABLED=0).")

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
