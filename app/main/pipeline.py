# app/main/pipeline.py
"""
El pipeline de respuesta: de una question a una respuesta guardada.

    override o router  ->  tool_calls  ->  un bucle de ejecución  ->  formato

Es el equivalente de `services/generation.py` en aviation-rag-service: la lógica
vive aquí y los endpoints solo la invocan. Antes estaba todo dentro de
`routes.py`, duplicado entre `ask()` y `edit_and_resubmit()`.
"""

import re

from flask import current_app, jsonify
from flask_login import current_user
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app import db
from app.models import Message
from app.rag_logic.agent_reasoning import ReasoningAgent
from app.rag_logic.agent_router import AgentRouter
from app.rag_logic.agent_intermedios import SQLAgent
from app.rag_logic.cost_calculator import calculate_cost
from app.rag_logic.excel_tool import ExcelAnalysisTool
from app.rag_logic.sql_tool import SQLDatabaseTool
from app.rag_logic.tools import ChatWithDocumentTool, SummarizeDocumentTool
from app.rag_logic.web_search import WebSearchTool


def clean_metadata_for_json(metadata):
    clean_meta = {}
    for key, value in (metadata or {}).items():
        if hasattr(value, "item"):
            clean_meta[key] = value.item()
        else:
            clean_meta[key] = value
    return clean_meta
def _settings_with_acl(base_settings, user) -> dict:
    """
    Inject the current user's department access guardrail into the settings
    passed to the document tools. Without it,
    qa_chain.get_conversational_qa_chain does not know what the user may see and,
    being fail-closed by design, would deny everything.
    """
    settings = dict(base_settings or {})
    settings["allowed_departments"] = user.get_allowed_departments()
    return settings
def _extract_user_mode(raw_text: str):
    """
    Detecta modo explícito del usuario (case-insensitive) SOLO si aparece AL INICIO:
      - "SQL"   -> fuerza ruta SQL
      - "AMBAS" -> fuerza ruta híbrida (SQL -> DOCS)

    Formatos soportados (ejemplos):
      - "SQL: dame el top 10..."
      - "SQL dame el top 10..."
      - "AMBAS - compara esto y dame context..."

    Devuelve: (mode, cleaned_text)
      - mode: "sql" | "ambas" | None
      - cleaned_text: question sin el prefijo detectado
    """
    if not raw_text:
        return None, raw_text

    text = raw_text.strip()
    # Solo si aparece al prefix como palabra completa.
    m = re.match(r"^(sql|ambas)\b", text, flags=re.IGNORECASE)
    if not m:
        return None, text

    mode = m.group(1).lower()
    cleaned = text[m.end():].strip()
    # Limpia separadores típicos tras el prefijo: "SQL: ..." | "SQL - ..." | "SQL — ..."
    cleaned = re.sub(r"^[:\-—\s,.;/]+", "", cleaned).strip()
    
    return mode, (cleaned or text)
def _make_chat_title_from_question(q: str, max_len: int = 46) -> str:
    q = (q or "").strip()
    if not q:
        return "New chat"

    # Limpieza rápida
    q = re.sub(r"\s+", " ", q)
    q = q.replace("\n", " ").strip()

    # Título estilo GPT: primeras palabras, sin hora
    # (si es muy largo, recortamos)
    title = q
    if len(title) > max_len:
        title = title[:max_len].rsplit(" ", 1)[0].strip() + "…"

    return title or "New chat"
class _ToolBox:
    """The project's tools, built once per request."""

    # Fixed collection identifier for the chain cache key. It used to be the
    # uuid of a `project` row; the application serves one knowledge base, so a
    # constant states the truth without a table behind it.
    COLLECTION = "kb"

    def __init__(self, model_name, user, logger, use_web_search=False):
        cfg = current_app.config
        settings = {
            "system_instruction": cfg.get("SYSTEM_INSTRUCTION", ""),
            "sql_context": cfg.get("SQL_CONTEXT", ""),
        }
        acl_settings = _settings_with_acl(settings, user)
        allowed = user.get_allowed_departments()
        vector_store_path = cfg["UP_VECTOR_STORE_PATH"]

        self.docs = ChatWithDocumentTool(
            project_id=self.COLLECTION,
            vector_store_path=vector_store_path,
            model_name=model_name,
            project_settings=acl_settings,
        )
        self.summary = SummarizeDocumentTool(
            project_id=self.COLLECTION,
            vector_store_path=vector_store_path,
            model_name=model_name,
            project_settings=acl_settings,
        )
        self.sql = SQLDatabaseTool(
            model_name=model_name,
            project_settings=settings,
            allowed_departments=allowed,
        )
        self.excel = ExcelAnalysisTool(
            doc_path=cfg["KNOWLEDGE_BASE_PATH"],
            model_name=model_name,
            allowed_departments=allowed,
        )
        self.web = WebSearchTool() if use_web_search else None

        self.all = [t for t in (self.docs, self.summary, self.sql, self.excel, self.web) if t is not None]

        self.logger = logger
        self.sql_agent = SQLAgent(self.sql, model_name=model_name, callbacks=[logger])
        self.reasoning = ReasoningAgent(model_name=model_name, callbacks=[logger])
def _sql_context_document(step_result):
    """Compress SQL output into a Document, to chain SQL → DOCS."""
    text = (step_result.get("sql_raw_output") or step_result.get("answer") or "").strip()
    if not text:
        return None
    compacto = "\n".join([ln for ln in text.splitlines() if ln.strip()][:25])
    return Document(
        page_content=f"[SALIDA SQL - RESUMEN]\n{compacto}",
        metadata={"source": "SQL", "type": "sql_context"},
    )
def _calls_from_override(raw_text, box):
    """
    Explicit user modes (SQL:, AMBAS, or asking for a summary) produce the SAME
    tool_calls structure the router would return.

    That is the simplification that removed three duplicated blocks: an override
    is not a different path, it only skips the decision. What runs afterwards is
    identical.

    Returns (calls, clean_question). calls=None means "let the router decide".
    """
    user_mode, cleaned = _extract_user_mode(raw_text)
    question = cleaned or raw_text
    qt = question.lower()

    if any(k in qt for k in ("resumen", "resume", "summary", "síntesis")):
        m = re.search(r"(?:del|de la|documento|pdf|pptx?|presentación)\s+([a-z0-9_\-\. ]+)", qt)
        hint = (m.group(1).strip() if m else "")
        return [{"name": box.summary.name, "args": {"doc_name_hint": hint}}], question

    if user_mode == "sql":
        return [{"name": box.sql.name, "args": {"query": question}}], question

    if user_mode == "doc":
        return [{"name": box.docs.name, "args": {"question": question}}], question

    if user_mode in ("ambas", "hib"):
        return [
            {"name": box.sql.name, "args": {"query": question}},
            {"name": box.docs.name, "args": {"question": question}},
        ], question

    return None, question
def _run_tools(calls, box, question, paired_history):
    """
    The single tool dispatch loop.

    Takes the tool_calls list from wherever it came — the router or an override —
    and returns normalised results. When a SQL call produces output, it is chained
    as context into the following document call (the hybrid mode).
    """
    results = []
    sql_context = None

    for call in calls or []:
        name = call.get("name")
        args = call.get("args") or {}

        if name == box.docs.name:
            q = args.get("question") or question
            # Context from a preceding SQL step is prepended to the question,
            # since the document chain takes no extra documents.
            if sql_context is not None:
                q = f"{q}\n\n{sql_context.page_content}"
            step = box.docs.run(
                {"question": q, "chat_history": paired_history},
                callbacks=[box.logger],
            )

        elif name == box.summary.name:
            step = box.summary.run(
                {"doc_name_hint": args.get("doc_name_hint", "") or ""},
                callbacks=[box.logger],
            )

        elif name == box.sql.name:
            step = box.sql_agent.run(args.get("query") or question)
            if isinstance(step, dict):
                sql_context = _sql_context_document(step)

        elif name == box.excel.name:
            step = box.excel.run(
                {
                    "query": args.get("query") or args.get("question") or question,
                    "file_name_hint": args.get("file_name_hint", "") or "",
                },
                callbacks=[box.logger],
            )

        elif box.web is not None and name == box.web.name:
            step = box.web.run({"query": args.get("query") or question}, callbacks=[box.logger])

        else:
            step = {"answer": f"No sé qué herramienta usar para: {name}", "source_documents": []}

        if not isinstance(step, dict):
            step = {"answer": str(step) if step else "Error interno en la herramienta.", "source_documents": []}
        step["origin"] = name
        results.append(step)

    return results
def _history_for(session, hasta=None, limit=10):
    """
    Conversation history in the two shapes needed: LangChain messages for the
    router, and (user, bot) pairs for the chain. `hasta` limits it to everything
    before a given message, for regeneration.
    """
    q = Message.query.filter_by(session_id=session.id, user_id=current_user.id)
    if hasta is not None:
        mensajes = q.filter(Message.timestamp < hasta).order_by(Message.timestamp.asc()).all()
    else:
        mensajes = q.order_by(Message.timestamp.desc()).limit(limit).all()
        mensajes.reverse()

    para_router = [
        (HumanMessage(content=m.content) if m.sender == "user" else AIMessage(content=m.content))
        for m in mensajes
    ]
    paired = [
        (mensajes[i].content, mensajes[i + 1].content)
        for i in range(0, len(mensajes) - 1, 2)
        if mensajes[i].sender == "user" and mensajes[i + 1].sender == "bot"
    ]
    return para_router, paired
def _answer_question(session, question, model_name, box,
                     historial_hasta=None, use_router=True):
    """
    The whole pipeline, and the only place it lives.

    override or router → tool_calls → one dispatch loop → formatting agent.
    Returns (final_result, question_for_the_title).
    """
    para_router, paired = _history_for(session, hasta=historial_hasta)

    calls, clean_question = _calls_from_override(question, box)

    if calls is None and use_router:
        router = AgentRouter(model_name=model_name, tools=box.all,
                             doc_path=current_app.config["KNOWLEDGE_BASE_PATH"])
        eleccion = router.route(question, para_router, callbacks=[box.logger])

        if not getattr(eleccion, "tool_calls", None):
            # Router path 1: it answers directly, with no tools and no
            # retrieval, so there are no sources to cite.
            crudo = (getattr(eleccion, "content", "") or "").strip()
            if crudo:
                primera, *resto = crudo.splitlines()
                if primera.strip().upper().startswith("ROUTE:"):
                    crudo = "\n".join(resto).strip()
            return {"answer": crudo or "No tengo respuesta para eso.", "source_documents": []}, clean_question

        calls = eleccion.tool_calls

    with get_openai_callback() as cb:
        results = _run_tools(calls, box, clean_question, paired)
    # El cost se registra por consulta. Antes se acumulaba en `project.cost`,
    # una columna que ninguna pantalla mostraba.
    cost = calculate_cost(model_name, cb.prompt_tokens, cb.completion_tokens)
    print(f"💶 Query cost: {cost:.6f} "
          f"({cb.prompt_tokens} prompt + {cb.completion_tokens} completion tokens)")

    if not results:
        results = [{"answer": "No se ejecutó ninguna herramienta.", "source_documents": [], "origin": box.docs.name}]

    return box.reasoning.run(clean_question, results), clean_question
def _finalize(session, question_text, final_result,
              title_question=None, existing_user_message=None):
    """
    Common close of a response: format sources, persist the message pair, rename
    the chat if it is still untitled, and return the JSON.

    This block was COPIED FIVE TIMES inside ask() and twice more in
    edit_and_resubmit(). Not merely ugly: that copy-paste produced the three
    `NameError: memory_store` crashes ruff exposed once F821 stopped being
    ignored — someone duplicated the block and dropped a line on the way.
    """
    answer_text = final_result.get("answer") or "Error generando respuesta."

    sources_formatted = []
    for doc in final_result.get("source_documents", []) or []:
        if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
            sources_formatted.append({
                "page_content": doc.page_content,
                "metadata": clean_metadata_for_json(doc.metadata),
            })
        elif isinstance(doc, dict):
            sources_formatted.append(doc)

    # On a regeneration the user message already exists — it was just edited —
    # so only the new answer is added.
    if existing_user_message is not None:
        user_msg = existing_user_message
    else:
        user_msg = Message(
            session_id=session.id, user_id=current_user.id,
            sender="user", content=question_text,
        )
        db.session.add(user_msg)

    bot_msg = Message(
        session_id=session.id, user_id=current_user.id,
        sender="bot", content=answer_text, sources=sources_formatted,
    )
    db.session.add(bot_msg)

    if (session.name or "").strip().lower() in ("nuevo chat", "new chat"):
        session.name = _make_chat_title_from_question(title_question or question_text)


    db.session.commit()

    return jsonify({
        "success": True,
        "answer": answer_text,
        "sources": sources_formatted,
        "user_message_id": user_msg.id,
        "bot_message_id": bot_msg.id,
    })
