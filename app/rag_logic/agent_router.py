# app/rag_logic/agent_router.py

import re
import unicodedata
from typing import List, Optional, Any, Dict

from .llm_factory import get_llm
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.schema import AIMessage


def _norm(s: str) -> str:
    """Minúsculas, sin acentos y sin espacios sobrantes."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _contains_whole_word(text_norm: str, key: str) -> bool:
    """
    ¿Aparece `key` como palabra completa en `text_norm`?

    `key in text` es SUBCADENA, y ese era el defecto: "suma" casaba dentro
    de "consumar", "file" dentro de "filete", "hoja" dentro de "hojalata".
    Peligroso justo aquí porque el camino forzado del router se salta el LLM:
    un falso positivo desvía la question a la herramienta equivocada sin que
    nada lo revise.

    Se usan `(?<!\\w)` y `(?!\\w)` en vez de `\\b` porque hay keys que
    empiezan por punto —".xlsx"— y `\\b` no se comporta igual delante de un
    carácter que no es de palabra.

    Las keys de varias palabras admiten espacios variables: "hoja de calculo"
    casa también con "hoja  de   calculo".

    ⚠️ Contrapartida asumida: al exigir palabra completa se pierden las
    variantes morfológicas. "empleado" ya no casa con "empleados". Por eso las
    listas de abajo enumeran las formas que interesan —"cobra"/"cobran",
    "empleados"— en vez de confiar en la coincidencia parcial. Es la decisión
    correcta: un falso negativo manda la question al LLM, que decide bien; un
    falso positivo la manda a la herramienta equivocada sin red.
    """
    key_norm = _norm(key)
    if not key_norm:
        return False

    # La guarda solo se pone donde hay frontera de palabra que proteger. Una
    # key como ".xlsx" va pegada al name del fichero ("ventas.xlsx"), así
    # que exigir que no la preceda un carácter de palabra la haría imposible
    # de encontrar. Se mira el primer y el último carácter de la clave.
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
    # Señales fuertes: el usuario habla explícitamente de un fichero Excel/spreadsheet.
    strong_signals = ["excel", ".xlsx", ".xls", "spreadsheet", "hoja de calculo", "hoja de cálculo"]
    if _any_whole_word(qn, strong_signals):
        return True

    # Señales débiles ("tabla", "hoja", "dashboard", "cells") son demasiado genéricas:
    # se usan igual para pedir results de SQL o resúmenes de docs. Solo cuentan
    # si además se menciona explícitamente un archivo, o si piden un cálculo sobre un archivo.
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
        if _any_whole_word(qn, ["help", "ayuda", "what can you do", "que puedes hacer",
                                "who are you", "quien eres"]):
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
        # (menos cost, más determinista). Si ADEMÁS parece pedir context de
        # documentos (políticas, procedimientos...), encadenamos SQL → DOCS
        # (modo híbrido ya soportado en routes.py vía sql_context_doc).
        if _looks_like_sql_intent(user_input):
            class _ForcedChoice:
                def __init__(self, tool_calls):
                    self.tool_calls = tool_calls
                    self.content = ""

            calls = [{"name": "query_hr_database", "args": {"query": user_input}}]
            if _looks_like_docs_intent(user_input):
                calls.append({"name": "chat_with_documents", "args": {"question": user_input}})
            return _ForcedChoice(calls)

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