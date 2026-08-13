# app/rag_logic/agent_sql.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Optional

from .llm_factory import get_llm
from langchain.schema import SystemMessage, HumanMessage

from .sql_tool import SQLDatabaseTool


ChatHistory = List[Tuple[str, str]]
AgentResult = Dict[str, Any]


def _normalize_result(raw: Union[str, AgentResult]) -> AgentResult:
    """
    Normalise any tool's output to a dict with `answer` and `source_documents`.
    """
    if isinstance(raw, str):
        return {"answer": raw, "source_documents": []}
    if "answer" not in raw:
        raw["answer"] = ""
    if "source_documents" not in raw:
        raw["source_documents"] = []
    return raw


def _build_context_from_docs(docs: List[Any], max_docs: int = 8, max_chars: int = 2000) -> str:
    """
    Build a readable context block from source_documents.

    Accepts both LangChain Documents and serialised dicts.
    """
    if not docs:
        return ""

    chunks = []
    for d in docs[:max_docs]:
        if hasattr(d, "page_content"):
            text = getattr(d, "page_content", "") or ""
            meta = getattr(d, "metadata", {}) or {}
        elif isinstance(d, dict):
            text = d.get("page_content", "") or ""
            meta = d.get("metadata", {}) or {}
        else:
            text, meta = "", {}

        text = text[:max_chars]
        fuente = meta.get("source") or meta.get("relative_path") or ""
        header = f"[Fuente: {fuente}]" if fuente else "[Fragmento sin nombre de archivo]"
        chunks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(chunks)


# ======================================================================
# DOCUMENT QA AGENT — REMOVED, with data behind it
# ======================================================================
# It called the tool once and re-worded the answer with the LLM. Measured with
# evaluation/evaluate_pipeline.py over the 21 RAG questions of the golden
# dataset, in a PAIRED comparison — same retrieval, same base answer:
#
#     metric               without     with        delta
#     context_precision      0.7220      0.7180    -0.0040   <- IDENTICAL context
#     context_recall         0.8095      0.8095    +0.0000   <- control holds
#     faithfulness           0.9382      0.9444    +0.0062
#     answer_relevancy       0.8327      0.8380    +0.0053
#
# context_precision moved -0.004 with identical context, which is impossible in
# principle: that is the judge's noise floor. The two metrics that could move did
# so within the same order of magnitude. The gain was indistinguishable from
# noise, and it cost +59% latency (1.39 s on top of a 2.35 s chain) plus one LLM
# call per query.
#
# The conversational memory (ChatMemoryStore) went with it: this was the only
# place `extra_documents` entered the context. Looked at closely, it stored only
# "the user asked: X" — questions, never answers — and re-injected them as
# retrieval context. That does not help answer anything: it adds noise and costs
# two embedding calls per query. Follow-up questions are unaffected; they go
# through chat_history and CONDENSE_QUESTION_PROMPT, a different mechanism.


# ======================================================================
# SUMMARY AGENT — REMOVED
# ======================================================================
# It was constructed on every /ask and every edit_and_resubmit, and nothing
# called it: both routes invoke `summary_tool.run(...)` directly. Its `run(self)`
# also took no arguments, unlike the other four agents, so "fixing" the call
# would have raised TypeError. Dead and broken. Summaries still come from
# SummarizeDocumentTool.


# ======================================================================
# SQL AGENT
# ======================================================================

class SQLAgent:
    """
    Queries the HR analytics database.

    It does NOT call the LLM, and that is deliberate: every consumer downstream
    reads `sql_raw_output` in preference to `answer`, so a rewriting pass here
    would be paid for and then discarded. The business-language reading is done
    by the SQL tool itself, and the final formatting by `ReasoningAgent`.
    """

    def __init__(
        self,
        tool: SQLDatabaseTool,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        callbacks: Optional[list] = None,
    ) -> None:
        # model_name/temperature stay in the signature so callers do not break,
        # but no LLM client is built: there is nobody to call.
        self.tool = tool
        self.name = tool.name
        self.callbacks = callbacks or []

    def run(self, query: str) -> AgentResult:
        raw = self.tool.run({"query": query}, callbacks=self.callbacks)
        result = _normalize_result(raw)

        sql_output = (result.get("answer") or "").strip()

        # The full data travels in sql_raw_output, which is what ReasoningAgent
        # and the hybrid SQL→DOCS chaining read.
        result["sql_raw_output"] = sql_output
        result["answer"] = sql_output
        return result


# ======================================================================
# EXCEL AGENT and WEB SEARCH AGENT — REMOVED
# ======================================================================
# Both did the same thing: call their tool ONCE and re-word the result with the
# LLM. They decided nothing, retried nothing and chose between nothing — they
# were post-processing decorators with a name that promised an agent.
#
# Each cost one LLM call per query to rewrite text that `ReasoningAgent` would
# reformat immediately afterwards: two presentation passes over the same content.
#
# The dispatch loop now calls the tools directly. Final presentation is
# ReasoningAgent's job, which is what it exists for.
