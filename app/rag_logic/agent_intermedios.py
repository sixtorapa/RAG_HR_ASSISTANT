# app/rag_logic/agent_intermedios.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Optional

from .llm_factory import get_llm
from langchain.schema import SystemMessage, HumanMessage

from .sql_tool import SQLDatabaseTool


ChatHistory = List[Tuple[str, str]]
AgentResult = Dict[str, Any]


def _normalize_result(raw: Union[str, AgentResult]) -> AgentResult:
    """
    Normaliza la salida de cualquier herramienta a un dict con:
    - answer: str
    - source_documents: list
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
    Construye un bloque de contexto legible a partir de los source_documents.
    Admite tanto Document de LangChain como dicts serializados.
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
# DOCUMENT QA AGENT — ELIMINADO (3-ago-2026, con datos)
# ======================================================================
# Llamaba a la tool una vez y volvía a redactar la respuesta con el LLM.
# Medido con evaluation/evaluate_pipeline.py sobre las 21 preguntas de RAG del
# golden dataset, en comparación PAREADA (mismo retrieval, misma respuesta base):
#
#     métrica              sin capa    con capa      delta
#     context_precision      0.7220      0.7180    -0.0040   <- contexto IDÉNTICO
#     context_recall         0.8095      0.8095    +0.0000   <- control correcto
#     faithfulness           0.9382      0.9444    +0.0062
#     answer_relevancy       0.8327      0.8380    +0.0053
#
# context_precision se movió -0.004 con el contexto idéntico, lo cual es
# imposible en principio: ése es el suelo de ruido del juez. Las dos métricas que
# sí podían moverse lo hicieron en ese mismo orden de magnitud. La mejora era
# indistinguible del ruido, y costaba +59% de latencia (1,39 s sobre 2,35 s de
# cadena) y una llamada al LLM por consulta, en un sistema que está a 2 segundos
# del techo de 29 s de API Gateway.
#
# Con él se fue la memoria conversacional (ChatMemoryStore): era el único sitio
# donde `extra_documents` entraba en el contexto. Y al mirarla de cerca guardaba
# solo "Usuario preguntó: X" —las preguntas, nunca las respuestas— y las
# reinyectaba como contexto de recuperación. Eso no ayuda a responder: añade
# ruido al contexto y cuesta dos llamadas de embeddings por consulta.
# Las repreguntas siguen funcionando: van por chat_history y
# CONDENSE_QUESTION_PROMPT, que es otro mecanismo y no se ha tocado.


# ======================================================================
# SUMMARY AGENT — ELIMINADO (29-jul → 1-ago 2026)
# ======================================================================
# Se construía en cada /ask y en cada edit_and_resubmit, y no lo llamaba
# nadie: las dos rutas invocan `summary_tool.run(...)` directamente. Además
# su `run(self)` no aceptaba argumentos, al contrario que los otros cuatro
# agentes, así que "arreglar" la llamada lo habría reventado con TypeError.
# Código muerto y además roto. El resumen lo sigue sirviendo SummarizeDocumentTool.


# ======================================================================
# SQL AGENT
# ======================================================================

class SQLAgent:
    """
    Agente de consultas a la BD analítica de RRHH.

    NO llama al LLM, y esto es deliberado. Antes hacía una pasada de
    reformulación en lenguaje de negocio, pero esa salida no la leía nadie:
    `ReasoningAgent._build_contributions_summary` y los dos consumidores de
    `routes.py` (829 y 998) toman `sql_raw_output` con preferencia sobre
    `answer`, así que el texto refinado se descartaba entero.

    Era una llamada al LLM por consulta SQL tirada a la basura, en un sistema
    cuyo cuello de botella medido son ~27 s contra el techo de 29 s de API
    Gateway. La interpretación en lenguaje de negocio ya la hace el propio
    `HRDatabaseTool` (paso 4 de su `_run`), y el formato final lo pone
    `ReasoningAgent`. Sobraba la del medio.
    """

    def __init__(
        self,
        tool: SQLDatabaseTool,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        callbacks: Optional[list] = None,
    ) -> None:
        # model_name/temperature se conservan en la firma para no romper a los
        # llamadores, pero ya no se construye cliente LLM: no hay a quién llamar.
        self.tool = tool
        self.name = tool.name
        self.callbacks = callbacks or []

    def run(self, query: str) -> AgentResult:
        raw = self.tool.run({"query": query}, callbacks=self.callbacks)
        result = _normalize_result(raw)

        sql_output = (result.get("answer") or "").strip()

        # El dato completo viaja en sql_raw_output, que es lo que leen
        # ReasoningAgent y el encadenado híbrido SQL→DOCS de routes.py.
        result["sql_raw_output"] = sql_output
        result["answer"] = sql_output
        return result


# ======================================================================
# EXCEL AGENT y WEB SEARCH AGENT — ELIMINADOS (1-ago-2026)
# ======================================================================
# Los dos hacían lo mismo: llamar a su tool UNA vez y volver a redactar el
# resultado con el LLM. Ni decidían, ni reintentaban, ni elegían entre nada:
# eran decoradores de post-proceso con un nombre que prometía un agente.
#
# Se pagaba una llamada al LLM por consulta para reescribir un texto que
# `ReasoningAgent` iba a reformatear inmediatamente después. Dos pasadas de
# presentación seguidas sobre el mismo contenido.
#
# routes.py llama ahora a `excel_tool` y `web_tool` directamente. La
# presentación final la hace ReasoningAgent, que es para lo que existe.
