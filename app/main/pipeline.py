# app/main/pipeline.py
"""
El pipeline de respuesta: de una pregunta a una respuesta guardada.

    override o router  ->  tool_calls  ->  un bucle de ejecución  ->  formato

Es el equivalente de `services/generation.py` en aviation-rag-service: la lógica
vive aquí y los endpoints solo la invocan. Antes estaba todo dentro de
`routes.py`, duplicado entre `ask()` y `edit_and_resubmit()`.
"""

import re

from flask import jsonify
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
    Inyecta el guardarril de control de acceso por departamento del usuario actual
    en los project_settings que se le pasan a ChatWithDocumentTool/SummarizeDocumentTool.
    Sin esto, qa_chain.get_conversational_qa_chain no sabe a qué tiene acceso el
    usuario y, por diseño fail-closed, denegaría todo (ver allowed_departments en
    qa_chain.py).
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
      - "AMBAS - compara esto y dame contexto..."

    Devuelve: (mode, cleaned_text)
      - mode: "sql" | "ambas" | None
      - cleaned_text: pregunta sin el prefijo detectado
    """
    if not raw_text:
        return None, raw_text

    text = raw_text.strip()
    # Solo si aparece al inicio como palabra completa.
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
        return "Nuevo chat"

    # Limpieza rápida
    q = re.sub(r"\s+", " ", q)
    q = q.replace("\n", " ").strip()

    # Título estilo GPT: primeras palabras, sin hora
    # (si es muy largo, recortamos)
    title = q
    if len(title) > max_len:
        title = title[:max_len].rsplit(" ", 1)[0].strip() + "…"

    return title or "Nuevo chat"
class _ToolBox:
    """Las herramientas del proyecto, construidas una sola vez por petición."""

    def __init__(self, project, model_name, user, logger, use_web_search=False):
        acl_settings = _settings_with_acl(project.settings, user)
        allowed = user.get_allowed_departments()

        self.docs = ChatWithDocumentTool(
            project_id=project.id,
            vector_store_path=project.vector_store_path,
            model_name=model_name,
            project_settings=acl_settings,
        )
        self.summary = SummarizeDocumentTool(
            project_id=project.id,
            vector_store_path=project.vector_store_path,
            model_name=model_name,
            project_settings=acl_settings,
        )
        self.sql = SQLDatabaseTool(
            model_name=model_name,
            project_settings=project.settings or {},
            allowed_departments=allowed,
        )
        self.excel = ExcelAnalysisTool(
            doc_path=project.document_path,
            model_name=model_name,
            allowed_departments=allowed,
        )
        self.web = WebSearchTool() if use_web_search else None

        self.all = [t for t in (self.docs, self.summary, self.sql, self.excel, self.web) if t is not None]

        self.logger = logger
        self.sql_agent = SQLAgent(self.sql, model_name=model_name, callbacks=[logger])
        self.reasoning = ReasoningAgent(model_name=model_name, callbacks=[logger])
def _sql_context_document(step_result):
    """Comprime la salida SQL en un Document para encadenar SQL → DOCS."""
    texto = (step_result.get("sql_raw_output") or step_result.get("answer") or "").strip()
    if not texto:
        return None
    compacto = "\n".join([ln for ln in texto.splitlines() if ln.strip()][:25])
    return Document(
        page_content=f"[SALIDA SQL - RESUMEN]\n{compacto}",
        metadata={"source": "SQL", "type": "sql_context"},
    )
def _calls_from_override(raw_text, box):
    """
    Los modos explícitos del usuario (@sql, @ambas, @doc, o pedir un "resumen")
    producen la MISMA estructura de tool_calls que devolvería el router.

    Esa es la simplificación que elimina tres bloques duplicados: un override no
    es un camino distinto, es solo saltarse la decisión. Lo que se ejecuta después
    es idéntico.

    Devuelve (calls, pregunta_limpia). calls=None significa "que decida el router".
    """
    user_mode, cleaned = _extract_user_mode(raw_text)
    pregunta = cleaned or raw_text
    qt = pregunta.lower()

    if any(k in qt for k in ("resumen", "resume", "summary", "síntesis")):
        m = re.search(r"(?:del|de la|documento|pdf|pptx?|presentación)\s+([a-z0-9_\-\. ]+)", qt)
        hint = (m.group(1).strip() if m else "")
        return [{"name": box.summary.name, "args": {"doc_name_hint": hint}}], pregunta

    if user_mode == "sql":
        return [{"name": box.sql.name, "args": {"query": pregunta}}], pregunta

    if user_mode == "doc":
        return [{"name": box.docs.name, "args": {"question": pregunta}}], pregunta

    if user_mode in ("ambas", "hib"):
        return [
            {"name": box.sql.name, "args": {"query": pregunta}},
            {"name": box.docs.name, "args": {"question": pregunta}},
        ], pregunta

    return None, pregunta
def _run_tools(calls, box, pregunta, paired_history):
    """
    Bucle ÚNICO de despacho de herramientas.

    Recibe la lista de tool_calls venga de donde venga —del router o de un
    override— y devuelve la lista de resultados normalizados. Si una llamada a
    SQL produce salida, se encadena como contexto a la siguiente de documentos
    (el modo híbrido).
    """
    resultados = []
    sql_context = None

    for llamada in calls or []:
        nombre = llamada.get("name")
        args = llamada.get("args") or {}

        if nombre == box.docs.name:
            q = args.get("question") or pregunta
            # El contexto de un paso SQL previo (modo híbrido) se antepone a la
            # pregunta, porque la cadena documental no acepta documentos extra.
            if sql_context is not None:
                q = f"{q}\n\n{sql_context.page_content}"
            paso = box.docs.run(
                {"question": q, "chat_history": paired_history},
                callbacks=[box.logger],
            )

        elif nombre == box.summary.name:
            paso = box.summary.run(
                {"doc_name_hint": args.get("doc_name_hint", "") or ""},
                callbacks=[box.logger],
            )

        elif nombre == box.sql.name:
            paso = box.sql_agent.run(args.get("query") or pregunta)
            if isinstance(paso, dict):
                sql_context = _sql_context_document(paso)

        elif nombre == box.excel.name:
            paso = box.excel.run(
                {
                    "query": args.get("query") or args.get("question") or pregunta,
                    "file_name_hint": args.get("file_name_hint", "") or "",
                },
                callbacks=[box.logger],
            )

        elif box.web is not None and nombre == box.web.name:
            paso = box.web.run({"query": args.get("query") or pregunta}, callbacks=[box.logger])

        else:
            paso = {"answer": f"No sé qué herramienta usar para: {nombre}", "source_documents": []}

        if not isinstance(paso, dict):
            paso = {"answer": str(paso) if paso else "Error interno en la herramienta.", "source_documents": []}
        paso["origin"] = nombre
        resultados.append(paso)

    return resultados
def _history_for(session, hasta=None, limite=10):
    """
    Historial de la conversación en los dos formatos que hacen falta:
    mensajes de LangChain para el router, y pares (usuario, bot) para la cadena.
    `hasta` acota a lo anterior a un mensaje concreto (regeneración).
    """
    q = Message.query.filter_by(session_id=session.id, user_id=current_user.id)
    if hasta is not None:
        mensajes = q.filter(Message.timestamp < hasta).order_by(Message.timestamp.asc()).all()
    else:
        mensajes = q.order_by(Message.timestamp.desc()).limit(limite).all()
        mensajes.reverse()

    para_router = [
        (HumanMessage(content=m.content) if m.sender == "user" else AIMessage(content=m.content))
        for m in mensajes
    ]
    emparejado = [
        (mensajes[i].content, mensajes[i + 1].content)
        for i in range(0, len(mensajes) - 1, 2)
        if mensajes[i].sender == "user" and mensajes[i + 1].sender == "bot"
    ]
    return para_router, emparejado
def _answer_question(session, project, pregunta, model_name, box,
                     historial_hasta=None, use_router=True):
    """
    El pipeline completo, y el único sitio donde vive.

    override o router → tool_calls → un bucle de ejecución → agente de formato.
    Devuelve (final_result, pregunta_para_el_titulo) o (None, ...) si el router
    respondió directamente sin herramientas — en ese caso `final_result` ya
    contiene la respuesta directa.
    """
    para_router, emparejado = _history_for(session, hasta=historial_hasta)

    calls, pregunta_limpia = _calls_from_override(pregunta, box)

    if calls is None and use_router:
        router = AgentRouter(model_name=model_name, tools=box.all, doc_path=project.document_path)
        eleccion = router.route(pregunta, para_router, callbacks=[box.logger])

        if not getattr(eleccion, "tool_calls", None):
            # Camino 1 del router: responde él mismo, sin herramientas y sin
            # recuperación, así que tampoco hay fuentes que citar.
            crudo = (getattr(eleccion, "content", "") or "").strip()
            if crudo:
                primera, *resto = crudo.splitlines()
                if primera.strip().upper().startswith("ROUTE:"):
                    crudo = "\n".join(resto).strip()
            return {"answer": crudo or "No tengo respuesta para eso.", "source_documents": []}, pregunta_limpia

        calls = eleccion.tool_calls

    with get_openai_callback() as cb:
        resultados = _run_tools(calls, box, pregunta_limpia, emparejado)
    project.cost += calculate_cost(model_name, cb.prompt_tokens, cb.completion_tokens)

    if not resultados:
        resultados = [{"answer": "No se ejecutó ninguna herramienta.", "source_documents": [], "origin": box.docs.name}]

    return box.reasoning.run(pregunta_limpia, resultados), pregunta_limpia
def _finalize(session, question_text, final_result,
              title_question=None, existing_user_message=None):
    """
    Cierre común de una respuesta: formatear fuentes, persistir el par de
    mensajes, renombrar el chat si hace falta, guardar el hecho en memoria y
    devolver el JSON.

    Este bloque estaba COPIADO CINCO VECES dentro de ask() (resumen, override
    SQL, override ambas, respuesta directa y camino principal) y dos más en
    edit_and_resubmit(). No es solo fealdad: de ese copia-pega salieron los tres
    `NameError: memory_store` que destapó ruff al dejar de ignorar F821 — alguien
    duplicó el bloque y se dejó una línea por el camino.
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

    # En una regeneración el mensaje del usuario ya existe (se acaba de editar),
    # así que solo se añade la respuesta nueva.
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
