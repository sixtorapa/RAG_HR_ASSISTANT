# app/rag_logic/agent_router.py

import re
from typing import List, Optional, Any, Dict

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.schema import AIMessage


def _norm(s: str) -> str:
    return (s or "").strip().lower()


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
        "headcount", "how many employees", "cuantos empleados", "empleados", "departments",
        "performance score", "rating", "top performers", "attrition", "turnover", "job postings"
    ]
    return any(k in qn for k in sql_signals)


def _looks_like_excel_intent(q: str) -> bool:
    qn = _norm(q)
    excel_signals = ["excel", ".xlsx", "spreadsheet", "sheet", "dashboard", "tabla", "hoja", "celdas"]
    # también permitir si pide sumas/cálculos y menciona “archivo” típico
    calc_signals = ["sum", "suma", "total", "promedio", "average", "median", "percent", "porcentaje"]
    file_signals = ["archivo", "fichero", "file"]
    return any(k in qn for k in excel_signals) or (any(k in qn for k in calc_signals) and any(k in qn for k in file_signals))


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

        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        self.llm_with_tools = llm.bind_tools(self.tools)

        # Importante: limitar “excel” a casos con señales claras.
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

5) EXCEL → use 'analista_de_excel'
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

        # 0) vacío / solo símbolos
        if not qn or re.fullmatch(r"[\W_]+", qn):
            return AIMessage(content="ROUTE: DIRECT — Empty/low-content message.\nHi! Ask me about HR policies (docs) or HR metrics (SQL).")

        # 1) Saludos / gracias
        if _is_greeting(q):
            return AIMessage(content="ROUTE: DIRECT — Greeting detected.\nHi! How can I help you — HR docs (policies) or HR data (SQL)?")
        if _is_thanks(q):
            return AIMessage(content="ROUTE: DIRECT — Acknowledgement.\nYou're welcome! What else can I help you with?")

        # 2) Meta / ayuda
        if any(k in qn for k in ["help", "ayuda", "what can you do", "que puedes hacer", "qué puedes hacer", "who are you", "quien eres", "quién eres"]):
            return AIMessage(
                content="ROUTE: DIRECT — Meta/help request.\nI can answer HR policy questions from internal docs, and HR metrics (salary, headcount, performance) from the HR database. What do you need?"
            )

        # 3) Smalltalk corto (evita LLM + tools en mensajes cortos)
        #    Regla: si <= 4 palabras y NO parece SQL/Excel/docs, responder directo.
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
        # ✅ Fast path para evitar que el LLM haga tonterías (Excel con “hi”)
        fast = self._fast_route(user_input)
        if fast is not None:
            return fast

        # ✅ Si parece SQL muy claro, forzamos tool_call SIN pasar por LLM
        # (menos coste, más determinista)
        if _looks_like_sql_intent(user_input):
            class _ForcedChoice:
                def __init__(self, tool_name: str, query: str):
                    self.tool_calls = [{"name": tool_name, "args": {"query": query}}]
                    self.content = ""
            return _ForcedChoice("query_hr_database", user_input)

        # ✅ Si parece Excel claro, forzamos Excel; si no, el LLM decide entre DOCS/SUMMARY/WEB
        if _looks_like_excel_intent(user_input):
            class _ForcedChoice:
                def __init__(self, tool_name: str, query: str):
                    self.tool_calls = [{"name": tool_name, "args": {"query": query, "file_name_hint": ""}}]
                    self.content = ""
            return _ForcedChoice("analista_de_excel", user_input)

        payload = {"input": user_input, "chat_history": chat_history}
        if callbacks:
            return self.router_chain.invoke(payload, config={"callbacks": callbacks})
        return self.router_chain.invoke(payload)