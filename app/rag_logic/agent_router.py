# app/rag_logic/agent_router.py

import re
import unicodedata
from typing import List, Optional, Any, Dict

from .llm_factory import get_llm
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.schema import AIMessage


def _norm(s: str) -> str:
    """Lowercase, accent-stripped, whitespace-trimmed."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _contains_whole_word(text_norm: str, key: str) -> bool:
    """
    Does `key` appear as a whole word inside `text_norm`?

    `key in text` is SUBSTRING containment, which matches "suma" inside
    "consumar", "file" inside "filete" and "hoja" inside "hojalata". That is
    dangerous precisely here, because the router's forced path skips the LLM: a
    false positive sends the question to the wrong tool with nothing reviewing it.

    `(?<!\\w)` and `(?!\\w)` are used instead of `\\b` because some keys start
    with a dot — ".xlsx" — and `\\b` behaves differently before a non-word
    character.

    Multi-word keys tolerate variable spacing: "hoja de calculo" also matches
    "hoja  de   calculo".

    ⚠️ Accepted trade-off: requiring whole words loses morphological variants.
    "empleado" no longer matches "empleados", so the lists below spell out the
    forms that matter rather than relying on partial matching. It is the right
    call: a false negative reaches the LLM, which decides well; a false positive
    reaches the wrong tool with no safety net.
    """
    key_norm = _norm(key)
    if not key_norm:
        return False

    # The guard only goes where there is a word boundary to protect. A key like
    # ".xlsx" sits against the filename ("sales.xlsx"), so demanding no preceding
    # word character would make it impossible to find. The first and last
    # characters of the key decide.
    prefix = r"(?<!\w)" if key_norm[0].isalnum() or key_norm[0] == "_" else ""
    suffix = r"(?!\w)" if key_norm[-1].isalnum() or key_norm[-1] == "_" else ""

    pattern = prefix + re.escape(key_norm).replace(r"\ ", r"\s+") + suffix
    return re.search(pattern, text_norm) is not None


def _any_whole_word(text_norm: str, keys: List[str]) -> bool:
    return any(_contains_whole_word(text_norm, k) for k in keys)


def _is_greeting(q: str) -> bool:
    qn = _norm(q)
    return bool(re.match(r"^(hi|hello|hey|hola|buenas|buenos dias|buenas tardes|buenas noches)\b", qn))


def _is_thanks(q: str) -> bool:
    qn = _norm(q)
    return bool(re.match(r"^(thanks|thank you|gracias|perfecto|genial|vale|ok)\b", qn))


def _looks_like_sql_intent(q: str) -> bool:
    qn = _norm(q)
    sql_signals = [
        "salary", "sueldo", "salario", "pay", "compensation", "highest paid", "lowest paid",
        "headcount", "how many employees", "cuantos empleados", "cuántos empleados", "empleados",
        "departments", "departamento", "departamentos",
        "performance score", "rating", "top performers", "attrition", "turnover", "job postings",
        "cobra", "cobran", "gana", "ganan", "nomina", "nómina", "antiguedad", "antigüedad"
    ]
    return _any_whole_word(qn, sql_signals)


def _looks_like_docs_intent(q: str) -> bool:
    qn = _norm(q)
    docs_signals = [
        "policy", "politica", "política", "procedure", "procedimiento", "handbook",
        "manual", "onboarding", "benefits", "beneficios", "normativa", "reglamento",
        "vacaciones", "permiso", "baja", "licencia"
    ]
    return _any_whole_word(qn, docs_signals)


def _looks_like_excel_intent(q: str) -> bool:
    qn = _norm(q)
    # Strong signals: the user explicitly names an Excel file or spreadsheet.
    strong_signals = ["excel", ".xlsx", ".xls", "spreadsheet", "hoja de calculo", "hoja de cálculo"]
    if _any_whole_word(qn, strong_signals):
        return True

    # Weak signals ("tabla", "hoja", "dashboard", "celdas") are too generic: they
    # serve SQL results and document summaries just as often. They only count
    # alongside an explicit file, or a calculation over a file.
    weak_signals = ["sheet", "dashboard", "tabla", "hoja", "cells"]
    calc_signals = ["sum", "suma", "total", "promedio", "average", "median", "percent", "porcentaje"]
    file_signals = ["archivo", "fichero", "file"]

    has_weak = _any_whole_word(qn, weak_signals)
    has_calc = _any_whole_word(qn, calc_signals)
    has_file = _any_whole_word(qn, file_signals)

    return (has_weak and has_file) or (has_calc and has_file)


class AgentRouter:
    """
    AGENT ORCHESTRATOR — HR Knowledge Base Assistant
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

        llm = get_llm(self.model_name, self.temperature)
        self.llm_with_tools = llm.bind_tools(self.tools)

        # Important: restrict Excel to cases with clear signals.
        system_prompt = f"""You are a ROUTING ORCHESTRATOR for an internal HR & Knowledge Base assistant.
Your ONLY job is to decide which tool(s) to call based on the user's question.

{extra_system_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1) DIRECT (no tools)
Use DIRECT ONLY for:
- Greetings / short smalltalk ("hi", "hola", "thanks")
- Meta questions ("what can you do?")

2) SQL  → use 'query_hr_database'
Use SQL when the user asks about:
- Salaries / compensation (highest/lowest/average)
- Headcount, counts, rankings, performance scores, attrition, job postings
IMPORTANT: user does NOT need to say "SQL".

3) DOCS → use 'chat_with_documents'
Use DOCS for:
- Policies, procedures, onboarding, benefits, handbook, any PDF/PPT content.

4) SUMMARISE → use 'summarise_document'
Only if user asks to summarise a document.

5) EXCEL → use 'excel_analyst'
CRITICAL RULE: NEVER call Excel unless the question explicitly mentions Excel/XLSX/sheet/dashboard
(or clearly asks to compute from an Excel file). Otherwise DO NOT use Excel.

MANDATORY FIRST LINE (always):
ROUTE: <route> — <reason in 8-15 words>
  <route> must be one of: DIRECT | DOCS | SQL | SUMMARISE | EXCEL | WEB | HYBRID(SQL→DOCS)
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        self.router_chain = prompt | self.llm_with_tools

    def _fast_route(self, user_input: str) -> Optional[AIMessage]:
        q = (user_input or "").strip()
        qn = _norm(q)

        # 0) Empty, or symbols only
        if not qn or re.fullmatch(r"[\W_]+", qn):
            return AIMessage(content="ROUTE: DIRECT — Empty/low-content message.\nHi! Ask me about HR policies (docs) or HR metrics (SQL).")

        # 1) Greetings and acknowledgements
        if _is_greeting(q):
            return AIMessage(content="ROUTE: DIRECT — Greeting detected.\nHi! How can I help you — HR docs (policies) or HR data (SQL)?")
        if _is_thanks(q):
            return AIMessage(content="ROUTE: DIRECT — Acknowledgement.\nYou're welcome! What else can I help you with?")

        # 2) Meta and help
        if _any_whole_word(qn, ["help", "ayuda", "what can you do", "que puedes hacer",
                                "who are you", "quien eres"]):
            return AIMessage(
                content="ROUTE: DIRECT — Meta/help request.\nI can answer HR policy questions from internal docs, and HR metrics (salary, headcount, performance) from the HR database. What do you need?"
            )

        # 3) Short smalltalk: four words or fewer with no SQL or Excel signal is
        #    answered directly, sparing an LLM call and a tool run.
        if len(qn.split()) <= 4 and not _looks_like_sql_intent(q) and not _looks_like_excel_intent(q):
            return AIMessage(
                content="ROUTE: DIRECT — Short smalltalk; no tools needed.\nGot it — tell me what you want to check (docs or HR data)."
            )

        return None

    def route(
        self,
        user_input: str,
        chat_history: List[Any],
        callbacks: Optional[List[Any]] = None,
    ):
        # Fast path: stops the LLM doing something silly, like Excel for "hi"
        fast = self._fast_route(user_input)
        if fast is not None:
            return fast

        # A clear SQL intent forces the tool call WITHOUT consulting the LLM:
        # less cost, more determinism. If the question also asks for document
        # context — policies, procedures — SQL and DOCS are chained.
        if _looks_like_sql_intent(user_input):
            class _ForcedChoice:
                def __init__(self, tool_calls):
                    self.tool_calls = tool_calls
                    self.content = ""

            calls = [{"name": "query_hr_database", "args": {"query": user_input}}]
            if _looks_like_docs_intent(user_input):
                calls.append({"name": "chat_with_documents", "args": {"question": user_input}})
            return _ForcedChoice(calls)

        # A clear Excel intent forces Excel; otherwise the LLM chooses.
        if _looks_like_excel_intent(user_input):
            class _ForcedChoice:
                def __init__(self, tool_name: str, query: str):
                    self.tool_calls = [{"name": tool_name, "args": {"query": query, "file_name_hint": ""}}]
                    self.content = ""
            return _ForcedChoice("excel_analyst", user_input)

        payload = {"input": user_input, "chat_history": chat_history}
        if callbacks:
            return self.router_chain.invoke(payload, config={"callbacks": callbacks})
        return self.router_chain.invoke(payload)