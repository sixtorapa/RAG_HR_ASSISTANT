# app/rag_logic/summarizer.py

from __future__ import annotations

from typing import List, Optional, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate


# ── Config ────────────────────────────────────────────────────────────────────

MAX_DOCS_PROJECT = 40   # when summarising the full knowledge base
MAX_DOCS_SINGLE  = 60   # when summarising a single document (by hint)
MIN_DOCS_SINGLE  = 15   # fallback minimum if hint matches few chunks
TEMPERATURE      = 0.2


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPT_MAP = PromptTemplate.from_template("""
You are an expert analyst. Summarise this fragment WITHOUT losing useful detail.

FRAGMENT:
{text}

Return between 6 and 12 bullet points. Rules:
- If figures, dates, percentages, rankings or comparisons appear → INCLUDE THEM.
- If there are definitions, methodology, segmentations or assumptions → INCLUDE THEM.
- If there are conclusions/implications → INCLUDE THEM as "Implication: ..."
- Avoid generic phrases like "the text talks about...". Be specific.

Output format (bullet points):
- Finding:
- Figure/Data (if applicable):
- Context/What it means:
- Implication (if applicable):
""".strip())

PROMPT_COMBINE = PromptTemplate.from_template("""
You are a senior consultant. Create a DETAILED SUMMARY (not a short one) integrating all fragments.

FRAGMENTS TO SUMMARISE:
{text}

IMPORTANT instructions:
- Do not invent data. If something does not appear, do not complete it.
- Prioritise quantitative content (figures/dates/%/rankings) and actionable insights.
- If there are different topics, group them into sections.
- The result should be substantially longer than a typical executive summary (aim for depth).

Return the summary in THIS FORMAT:

## DETAILED SUMMARY

### 1) What the document is about
- (3 to 6 specific bullet points)

### 2) Structure / main topic blocks detected
- Topic/Block 1: (1-2 lines)
- Topic/Block 2: (1-2 lines)
- Topic/Block 3: (1-2 lines)
- (add more if applicable)

### 3) Key findings (detailed)
- (12 to 20 bullet points, each with detail; include figures where available)

### 4) Relevant data and metrics
- Metric / Data: value | context | (if applicable) comparison / trend
- (minimum 5 items if data is present)

### 5) Implications and actionable insights
1. ...
2. ...
3. ...
4. ...
5. ...
(between 5 and 10 implications)

### 6) Recommendations
- (6 to 12 concrete, applicable recommendations aligned with the content)

### 7) Notes / limitations / assumptions (if present)
- ...

### 8) Quick glossary (optional)
- Term: brief definition
(Only if the document introduces specific jargon or proprietary concepts)
""".strip())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meta_text(meta: dict) -> str:
    return " ".join(
        str(meta.get(k, "") or "")
        for k in ("filename", "relative_path", "source", "folder_context")
    ).lower()


def _decorate_docs_with_source(docs: List[Any]) -> List[Any]:
    """Injects metadata into page_content so the LLM keeps source references."""
    out = []
    for d in docs:
        try:
            meta  = d.metadata or {}
            fname = meta.get("filename") or meta.get("source") or meta.get("relative_path") or "document"
            page  = meta.get("page", None)
            slide = meta.get("slide", None)

            locator = ""
            if slide is not None:
                locator = f" | slide {int(slide)}"
            elif page is not None:
                try:
                    locator = f" | p. {int(page) + 1}"
                except Exception:
                    locator = f" | p. {page}"

            header = f"[SOURCE: {fname}{locator}]\n"
            d.page_content = header + (d.page_content or "").strip()
        except Exception:
            pass
        out.append(d)
    return out


def _downsample_evenly(docs: List[Any], max_docs: int) -> List[Any]:
    """Evenly samples chunks across the document to avoid only reading the beginning."""
    n = len(docs)
    if n <= max_docs:
        return docs

    step    = max(1, n // max_docs)
    sampled = docs[::step][:max_docs]

    if len(sampled) < max_docs:
        tail_needed = max_docs - len(sampled)
        sampled.extend(docs[-tail_needed:])

    return sampled[:max_docs]


# ── Main function ─────────────────────────────────────────────────────────────

def resumir_documentos_proyecto(
    vector_store_path: str,
    nombre_modelo: str,
    descripcion_proyecto: str = "",
    doc_name_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a DETAILED summary of the knowledge base or a specific document (via doc_name_hint).

    Returns:
        dict with 'texto_salida' (summary text) and 'documentos_fuente' (Document list)
    """
    try:
        print(f"Starting summary for: {vector_store_path}")

        embeddings    = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store  = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        docs: List[Any] = []

        # 1) Specific document requested via hint
        if doc_name_hint:
            hint = doc_name_hint.lower().strip()
            print(f"🔎 Summary scoped to document: {doc_name_hint}")

            raw      = vector_store.get(include=["documents", "metadatas"])
            all_docs = []
            for txt, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
                meta = meta or {}
                if hint and hint in _meta_text(meta):
                    from langchain_core.documents import Document
                    all_docs.append(Document(page_content=txt or "", metadata=meta))

            if all_docs:
                all_docs = _decorate_docs_with_source(all_docs)
                docs     = _downsample_evenly(all_docs, MAX_DOCS_SINGLE)

            # Fallback: similarity search if metadata match fails
            if not docs:
                docs = vector_store.similarity_search(
                    query=f"summary of document {doc_name_hint}",
                    k=max(MIN_DOCS_SINGLE, min(MAX_DOCS_SINGLE, 30)),
                )
                docs = _decorate_docs_with_source(docs)

        # 2) Full knowledge base summary
        else:
            docs = vector_store.similarity_search(
                query="detailed summary overview main topics key findings conclusions figures",
                k=MAX_DOCS_PROJECT,
            )
            docs = _decorate_docs_with_source(docs)

        if not docs:
            return {"texto_salida": "No documents found to summarise.", "documentos_fuente": []}

        llm = ChatOpenAI(model_name=nombre_modelo, temperature=TEMPERATURE)

        # If a project description is provided, use a contextualised map prompt
        if descripcion_proyecto:
            prompt_map_final = PromptTemplate.from_template(f"""
You are an expert analyst. Extract information especially relevant to this objective:
{descripcion_proyecto}

FRAGMENT:
{{text}}

Return between 6 and 12 bullet points:
- Finding:
- Figure/Data (if applicable):
- Context/What it means:
- Implication (if applicable):
""".strip())
        else:
            prompt_map_final = PROMPT_MAP

        summarise_chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=prompt_map_final,
            combine_prompt=PROMPT_COMBINE,
            verbose=True,
        )

        print(f"Summarising {len(docs)} fragments.")
        result = summarise_chain.invoke(docs)

        return {
            "texto_salida":     result.get("output_text", "").strip(),
            "documentos_fuente": docs,
        }

    except Exception as e:
        print(f"Error in summarizer: {e}")
        return {"texto_salida": f"Error generating summary: {str(e)}", "documentos_fuente": []}


# ── Legacy alias ──────────────────────────────────────────────────────────────

def summarize_project_documents(vector_store_path: str, model_name: str):
    """[DEPRECATED] Use resumir_documentos_proyecto() instead. Kept for backwards compatibility."""
    result = resumir_documentos_proyecto(vector_store_path, model_name)
    return {"output_text": result["texto_salida"], "source_documents": result["documentos_fuente"]}
