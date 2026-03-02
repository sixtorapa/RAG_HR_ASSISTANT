# app/rag_logic/agent_router.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool


# =========================
# Helpers (deterministas)
# =========================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _is_smalltalk(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return True
    # 1-2 palabras tipo "hi", "ok", "thanks"
    if len(qn.split()) <= 2 and re.match(r"^(hi|hello|hey|hola|buenas|ok|vale|gracias|thanks|bye|adios)\b", qn):
        return True
    # frases cortas típicas
    if re.match(r"^(hi|hello|hey|hola)[!. ]*$", qn):
        return True
    if re.match(r"^(thanks|gracias|thank you)[!. ]*$", qn):
        return True
    return False


def _looks_like_sql(q: str) -> bool:
    """
    Heurística: pregunta de HR cuantitativa / agregaciones / rankings.
    """
    qn = _norm(q)

    # palabras clave de números / ranking
    sql_markers = [
        "salary", "salaries", "compensation", "pay", "bonus",
        "highest", "lowest", "top", "bottom", "average", "avg",
        "count", "how many", "headcount", "total", "sum",
        "department", "departments", "by department",
        "attrition", "tenure", "hired", "terminated",
        "performance", "score", "rating", "reviews",
        "list", "rank", "ranking",
    ]

    # Si menciona entidades típicas del schema (employees, departments, etc.)
    schema_terms = ["employees", "employee", "departments", "performance_reviews", "job_postings"]

    if any(k in qn for k in sql_markers):
        return True
    if any(t in qn for t in schema_terms) and any(w in qn for w in ["highest", "lowest", "average", "count", "top", "rank"]):
        return True

    # Si hay números + pregunta HR (p.ej. "2024 performance top 10")
    if re.search(r"\b\d{4}\b", qn) and any(w in qn for w in ["top", "rank", "average", "count", "salary", "performance"]):
        return True

    return False


def _looks_like_excel(q: str) -> bool:
    """
    Heurística: cálculos típicos de Excel / tablas / dashboards.
    """
    qn = _norm(q)
    excel_markers = [
        "xlsx", "excel", "spreadsheet",
        "dashboard", "table", "tab",
        "sum", "total", "pivot",
        "what is the value", "how much",
        "variance", "difference", "vs",
    ]
    # ojo: muchas de estas también podrían ser SQL, pero Excel suele venir con “xlsx/excel/dashboard”
    if any(k in qn for k in ["xlsx", "excel", "spreadsheet"]):
        return True
    if "dashboard" in qn and any(k in qn for k in ["xlsx", "excel"]):
        return True
    if any(k in qn for k in excel_markers) and ("file" in qn or "sheet" in qn or "tab" in qn):
        return True
    return False


@dataclass
class _RouterChoice:
    """
    Objeto mínimo compatible con routes.py:
    - .content (str)
    - .tool_calls (list[dict] o None)
    """
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None


class AgentRouter:
    """
    Router híbrido:
    1) Fast-routing determinista (saludos, SQL claro, Excel claro)
    2) Si no es obvio, router LLM con tools
    """

    def __init__(
        self,
        model_name: str,
        tools: List[BaseTool],
        doc_path: Optional[str] = None,
        temperature: float = 0.0,
        extra_system_context: str = "",
    ) -> None:
        self.model_name = model_name
        self.tools = tools
        self.doc_path = doc_path
        self.temperature = temperature

        self._tool_by_name = {t.name: t for t in (tools or [])}

        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self.llm_with_tools = llm.bind_tools(self.tools)

        # Importante: menciona nombres REALES de herramientas (incluyendo Excel)
        excel_tool_name = "analista_de_excel" if "analista_de_excel" in self._tool_by_name else None
        web_tool_name = "web_search" if "web_search" in self._tool_by_name else None

        system_prompt = f"""You are a ROUTING ORCHESTRATOR for an internal HR assistant.
Your ONLY job is to decide which tool(s) to call based on the user's question.

{extra_system_context}

TOOLS (exact names):
- chat_with_documents
- summarise_document
- query_hr_database
{("- " + excel_tool_name) if excel_tool_name else ""}
{("- " + web_tool_name) if web_tool_name else ""}

ROUTING GUIDE:
A) DIRECT (no tool) ONLY for: greetings, thanks, short meta ("what can you do?").
B) SQL → query_hr_database for ANY quantitative HR question:
   salaries, headcount, rankings, averages, performance scores, attrition, tenure, job postings counts.
C) EXCEL → {excel_tool_name or "analista_de_excel"} when user refers to Excel/XLSX dashboards, sheets, tabs, table calculations.
D) DOCS → chat_with_documents for policies/processes/handbooks (PDF/PPTX/XLSX narrative docs).
E) SUMMARISE → summarise_document when user asks for a summary/overview.
F) WEB → web_search only if tool exists and user needs external live info.

MANDATORY:
- If the user asks about salaries, headcount, rankings, averages → choose SQL (even if they don't say SQL).
- Never invent data; route to a tool when needed.

Return tool calls when needed, otherwise a brief direct answer.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        self.router_chain = prompt | self.llm_with_tools

    # -----------------------
    # Deterministic pre-router
    # -----------------------
    def _choose_tool_name(self, preferred: List[str]) -> Optional[str]:
        for name in preferred:
            if name in self._tool_by_name:
                return name
        return None

    def _fast_route(self, user_input: str) -> Optional[_RouterChoice]:
        q = (user_input or "").strip()
        qn = _norm(q)

        # 1) Smalltalk => DIRECT
        if _is_smalltalk(q):
            # Respuesta breve, humana, sin tool.
            if re.match(r"^(thanks|gracias)\b", qn):
                return _RouterChoice(content="You're welcome! What would you like to check in HR docs or data?")
            if re.match(r"^(bye|adios)\b", qn):
                return _RouterChoice(content="Bye! If you need anything else, just ask.")
            return _RouterChoice(content="Hi! How can I help — HR policies (docs) or employee data (SQL)?")

        # 2) SQL claro
        if _looks_like_sql(q):
            sql_name = self._choose_tool_name(["query_hr_database"])
            if sql_name:
                return _RouterChoice(
                    content="",
                    tool_calls=[{"name": sql_name, "args": {"query": q}}],
                )

        # 3) Excel claro (solo si existe tool)
        if _looks_like_excel(q):
            excel_name = self._choose_tool_name(["analista_de_excel"])
            if excel_name:
                return _RouterChoice(
                    content="",
                    tool_calls=[{"name": excel_name, "args": {"query": q, "file_name_hint": ""}}],
                )

        return None

    # -----------------------
    # Main entrypoint
    # -----------------------
    def route(
        self,
        user_input: str,
        chat_history: List[Any],
        callbacks: Optional[List[Any]] = None,
    ):
        # 0) Fast route
        fr = self._fast_route(user_input)
        if fr is not None:
            return fr

        # 1) LLM route
        payload = {"input": user_input, "chat_history": chat_history}
        if callbacks:
            return self.router_chain.invoke(payload, config={"callbacks": callbacks})
        return self.router_chain.invoke(payload)