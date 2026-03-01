# app/main/routes.py

import os
import re
from datetime import datetime, timedelta

from flask import (
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    current_app,
    session as flask_session,
    abort,
    Response,
    stream_with_context,
)

from langchain_core.documents import Document

from flask_login import login_user, logout_user, login_required, current_user

from app import db
from app.main import bp

# Modelos
from app.models import ChatSession, Message, Project, User, LoginSession

# Lógica RAG / herramientas
from app.rag_logic.cost_calculator import calculate_cost
from app.rag_logic.excel_tool import ExcelAnalysisTool
from app.rag_logic.ingester import process_and_store_documents
from app.rag_logic.prompt_templates import get_all_templates, get_template_prompt
from app.rag_logic.qa_chain import chain_cache
from app.rag_logic.sql_tool import SQLDatabaseTool
from app.rag_logic.tools import ChatWithDocumentTool, SummarizeDocumentTool
from app.rag_logic.web_search import WebSearchTool

# Logger visual
from app.rag_logic.console_logger import ConsoleLogger

# Agentes
from app.rag_logic.agent_intermedios import (
    DocumentQAAgent,
    ExcelAgent,
    SQLAgent,
    SummaryAgent,
    WebSearchAgent,
)
from app.rag_logic.agent_reasoning import ReasoningAgent
from app.rag_logic.agent_router import AgentRouter

from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage

from app.rag_logic.chat_memory import ChatMemoryStore

# --- SANITIZACIÓN API KEY ---
raw_key = os.environ.get("OPENAI_API_KEY", "")
if raw_key:
    clean_key = raw_key.strip().strip("'").strip('"')
    os.environ["OPENAI_API_KEY"] = clean_key
    print(f"--- 🔑 API KEY SANITIZADA: {clean_key[:5]}... (Lista para usar) ---")


def clean_metadata_for_json(metadata):
    clean_meta = {}
    for key, value in (metadata or {}).items():
        if hasattr(value, "item"):
            clean_meta[key] = value.item()
        else:
            clean_meta[key] = value
    return clean_meta


def _get_or_create_single_project() -> Project:
    """
    SINGLE PROJECT legacy-safe:
    Prioridad de selección:
      1) Project con vector_store_path == UP_VECTOR_STORE_PATH (si existe)
      2) Project con name == HR Knowledge Base (evita UNIQUE name al crear)
      3) Primer Project existente
      4) Si no hay ninguno, crear uno nuevo
    """
    cfg = current_app.config

    name = (cfg.get("UP_PROJECT_NAME") or "HR Knowledge Base").strip()
    doc_path = (cfg.get("KNOWLEDGE_BASE_PATH") or "").strip()
    vector_path = (cfg.get("UP_VECTOR_STORE_PATH") or "").strip()

    target_vs = (vector_path or "").replace("\\", "/").strip()

    project = None

    # 1) Buscar por vector_store_path
    for p in Project.query.all():
        p_vs = (p.vector_store_path or "").replace("\\", "/").strip()
        if p_vs == target_vs and target_vs:
            project = p
            break

    # 2) Buscar por name
    if project is None and name:
        project = Project.query.filter_by(name=name).first()

    # 3) Primer project
    if project is None:
        project = Project.query.first()

    # 4) Crear
    if project is None:
        project = Project(
            name=name,
            document_path=doc_path or "-",
            vector_store_path=vector_path,
            status="READY",
            model_name="gpt-4o",
            settings={},
        )
        db.session.add(project)
        db.session.commit()
        return project

    # Normalización segura
    dirty = False

    if name and project.name != name:
        project.name = name
        dirty = True

    if doc_path and project.document_path != doc_path:
        project.document_path = doc_path
        dirty = True

    if project.status != "READY":
        project.status = "READY"
        dirty = True

    if project.settings is None:
        project.settings = {}
        dirty = True

    # Actualizar vector_store_path SOLO si no rompe UNIQUE
    cur_vs = (project.vector_store_path or "").replace("\\", "/").strip()
    if target_vs and cur_vs != target_vs:
        conflict = None
        for p in Project.query.all():
            if p.id == project.id:
                continue
            p_vs = (p.vector_store_path or "").replace("\\", "/").strip()
            if p_vs == target_vs:
                conflict = p
                break
        if conflict is None:
            project.vector_store_path = vector_path
            dirty = True

    if dirty:
        db.session.commit()

    return project


# ==================== API DE CHAT (CORE) ====================

def _clear_chain_cache_for_project(project_id: str) -> None:
    """En qa_chain.py la key del cache es: f"{project_id}::{model_name}" """
    try:
        prefix = f"{project_id}::"
        for k in list(chain_cache.keys()):
            if str(k).startswith(prefix):
                del chain_cache[k]
    except Exception:
        pass


def _bump_login_session_question() -> None:
    """
    Incrementa n_questions de la sesión de login activa (si existe).
    No rompe si falta (p.ej. sesión antigua / key inexistente).
    """
    try:
        ls_id = flask_session.get("login_session_id")
        if not ls_id:
            return

        ls = LoginSession.query.filter_by(id=ls_id, user_id=current_user.id).first()
        if not ls:
            return

        ls.n_questions = (ls.n_questions or 0) + 1
        ls.last_activity_at = datetime.utcnow()
        db.session.commit()
    except Exception:
        # Silencioso: esto nunca debe tumbar /ask
        try:
            db.session.rollback()
        except Exception:
            pass




# ==================== HOME (SINGLE APP) ====================

@bp.route("/")
@login_required
def index():
    """Pantalla única (Home + Asistente UP) sin gestión de proyectos."""
    project = _get_or_create_single_project()

    tab = (request.args.get("tab") or "home").lower().strip()
    if tab not in ("home", "chat", "activity"):
        tab = "home"

    # Si no es admin, activity no está permitido
    if tab == "activity" and getattr(current_user, "role", None) != "admin":
        tab = "home"


    session_id = (request.args.get("session") or "").strip()
    session = None

    # Si viene session id, solo si es del usuario
    if session_id:
        session = ChatSession.query.filter_by(
            id=session_id,
            project_id=project.id,
            user_id=current_user.id,
        ).first()

    # Lista de sesiones del usuario (para sidebar tipo ChatGPT)
    sessions_list = (
        ChatSession.query
        .filter_by(project_id=project.id, user_id=current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )

    # Si no viene session_id válido, usamos la última del usuario
    if session is None:
        session = sessions_list[0] if sessions_list else None

    # Si no hay ninguna, crear una nueva (IMPORTANTE: add+commit para evitar Detached)
    if session is None:
        session = ChatSession(
            name="Nuevo chat",
            project=project,
            user_id=current_user.id,
        )
        db.session.add(session)
        db.session.commit()
        sessions_list = [session]  # opcional: para que aparezca inmediatamente

    # ✅ Evitar lazy-load sobre session.messages (evita DetachedInstanceError)
    messages = (
        Message.query
        .filter_by(session_id=session.id, user_id=current_user.id)
        .order_by(Message.timestamp.asc())
        .all()
    )


    activity_sessions = []
    admin_activity = None  # <-- resumen + series para dashboard (solo admin)

    if tab == "activity" and getattr(current_user, "role", None) == "admin":
        # Tabla (detalle): últimas 200 sesiones
        activity_sessions = (
            db.session.query(LoginSession, User)
            .join(User, User.id == LoginSession.user_id)
            .order_by(LoginSession.started_at.desc())
            .limit(200)
            .all()
        )

        # Dashboard (resumen): últimos 30 días
        since = datetime.utcnow() - timedelta(days=30)

        last_30 = (
            db.session.query(LoginSession, User)
            .join(User, User.id == LoginSession.user_id)
            .filter(LoginSession.started_at >= since)
            .order_by(LoginSession.started_at.asc())
            .all()
        )

        total_sessions_30d = len(last_30)
        active_users_30d = len({u.id for (_, u) in last_30}) if last_30 else 0
        total_questions_30d = sum(int(ls.n_questions or 0) for (ls, _) in last_30) if last_30 else 0

        durations = [int(ls.duration_sec) for (ls, _) in last_30 if ls.duration_sec is not None]
        avg_duration_sec_30d = int(sum(durations) / len(durations)) if durations else 0

        # Series: sesiones por día
        by_day = {}
        for (ls, _) in last_30:
            key = (ls.started_at.date().isoformat() if ls.started_at else None)
            if not key:
                continue
            by_day[key] = by_day.get(key, 0) + 1

        days = sorted(by_day.keys())
        sessions_per_day = [by_day[d] for d in days]

        # Ranking: top usuarios por preguntas (y sesiones)
        per_user = {}
        for (ls, u) in last_30:
            item = per_user.get(u.username) or {"sessions": 0, "questions": 0}
            item["sessions"] += 1
            item["questions"] += int(ls.n_questions or 0)
            per_user[u.username] = item

        top_by_questions = sorted(
            [{"username": k, **v} for k, v in per_user.items()],
            key=lambda x: (x["questions"], x["sessions"]),
            reverse=True,
        )[:8]

        admin_activity = {
            "kpis": {
                "total_sessions_30d": total_sessions_30d,
                "active_users_30d": active_users_30d,
                "total_questions_30d": total_questions_30d,
                "avg_duration_sec_30d": avg_duration_sec_30d,
            },
            "series": {
                "days": days,
                "sessions_per_day": sessions_per_day,
            },
            "top_users": top_by_questions,
        }

    return render_template(
        "dashboard.html",
        project=project,
        session=session,
        messages=messages,
        sessions=sessions_list,
        active_tab=tab,
        activity_sessions=activity_sessions,
        admin_activity=admin_activity,  # <-- NUEVO
    )




@bp.route("/reindex", methods=["POST"])
def reindex():
    """
    Reindexa el único vector store:
    KNOWLEDGE_BASE_PATH -> UP_VECTOR_STORE_PATH (por defecto: vector_store/info)
    """
    # Protección simple: token por header o querystring
    token_cfg = (current_app.config.get("UP_ADMIN_TOKEN") or "").strip()
    token_in = (request.headers.get("X-UP-ADMIN-TOKEN") or request.args.get("token") or "").strip()
    if token_cfg and token_in != token_cfg:
        return ("Forbidden", 403)

    cfg = current_app.config
    doc_path = (cfg.get("KNOWLEDGE_BASE_PATH") or "").strip()
    vector_path = (cfg.get("UP_VECTOR_STORE_PATH") or "").strip()

    if not doc_path:
        flash("KNOWLEDGE_BASE_PATH is empty. Revisa config/.env.", "danger")
        return redirect(url_for("main.index"))

    if not os.path.exists(doc_path):
        flash(f"KNOWLEDGE_BASE_PATH no existe: {doc_path}", "danger")
        return redirect(url_for("main.index"))

    try:
        os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    except Exception as e:
        flash(f"No se pudo crear la carpeta de vector_store: {e}", "danger")
        return redirect(url_for("main.index"))

    try:
        ok = process_and_store_documents(doc_path, vector_path)
    except Exception as e:
        flash(f"Error reindexando: {e}", "danger")
        return redirect(url_for("main.index"))

    try:
        chain_cache.clear()
    except Exception:
        pass

    if ok:
        flash("✅ Índice actualizado correctamente.", "success")
    else:
        flash("⚠️ No se indexaron documentos (carpeta vacía o sin texto).", "warning")

    return redirect(url_for("main.index"))


# (Legacy) endpoints de gestión de proyectos -> ya no se usan
@bp.route("/create_project", methods=["POST"])
def create_project():
    flash("Esta app ya no usa proyectos. Usa Home / Asistente UP.", "warning")
    return redirect(url_for("main.index"))


@bp.route("/delete_project/<project_id>", methods=["POST"])
def delete_project(project_id):
    flash("Esta app ya no permite borrar proyectos. (SINGLE PROJECT).", "warning")
    return redirect(url_for("main.index"))


@bp.route("/edit_project/<project_id>", methods=["POST"])
def edit_project(project_id):
    flash("Esta app ya no permite editar proyectos. (SINGLE PROJECT).", "warning")
    return redirect(url_for("main.index"))


# ==================== GESTIÓN DE SESIONES ====================

@bp.route("/project/<project_id>")
@login_required
def project_overview(project_id):
    project = Project.query.get_or_404(project_id)
    sessions = (
        ChatSession.query
        .filter_by(project_id=project.id, user_id=current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return render_template("project_overview.html", project=project, sessions=sessions)


@bp.route("/create_chat", methods=["POST"])
@login_required
def create_chat():
    project = _get_or_create_single_project()
    new_session = ChatSession(name="Nuevo chat", project=project, user_id=current_user.id)

    db.session.add(new_session)
    db.session.commit()
    return redirect(url_for("main.index", tab="chat", session=new_session.id))


@bp.route("/delete_chat/<session_id>", methods=["POST"])
@login_required
def delete_chat(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    project_id = session.project_id

    Message.query.filter_by(session_id=session.id, user_id=current_user.id).delete(synchronize_session=False)


    db.session.delete(session)
    db.session.commit()
    _clear_chain_cache_for_project(project_id)

    flash("Chat eliminado.", "success")
    return redirect(url_for("main.index", tab="chat"))



@bp.route("/chat/<session_id>")
@login_required
def chat_session(session_id):
    # Legacy: redirige al shell único
    return redirect(url_for("main.index", tab="chat", session=session_id))


# ==================== API DE CHAT (CORE) ====================


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




@bp.route("/ask/<session_id>", methods=["POST"])
@login_required
def ask(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    project = session.project

    # ==================== MEMORIA CONVERSACIONAL ====================
    memory_store = ChatMemoryStore(
        persist_path=os.path.join(
            current_app.instance_path,
            "chat_memory",
            str(current_user.id)
        )
    )

    payload = request.get_json(silent=True) or {}
    question_text = payload.get("question")
    use_web_search = payload.get("use_web_search", False)
    selected_model = payload.get("model_name") or project.model_name or "gpt-4o"


    if not question_text:
        return jsonify({"error": "Falta la pregunta."}), 400

    # Observación básica: contar preguntas en la sesión de login
    _bump_login_session_question()

    debug_logger = ConsoleLogger()


    try:
        # Herramientas base
        chat_tool = ChatWithDocumentTool(
            project_id=project.id,
            vector_store_path=project.vector_store_path,
            model_name=selected_model,
            project_settings=project.settings or {},
        )
        summary_tool = SummarizeDocumentTool(
            project_id=project.id,
            vector_store_path=project.vector_store_path,
            model_name=selected_model,
        )
        sql_tool = SQLDatabaseTool(
            model_name=selected_model,
            project_settings=project.settings or {},
        )
        excel_tool = ExcelAnalysisTool(
            doc_path=project.document_path,
            model_name=selected_model,
        )

        tools = [chat_tool, summary_tool, sql_tool, excel_tool]

        web_tool = None
        if use_web_search:
            web_tool = WebSearchTool()
            tools.append(web_tool)

    

        # Agentes
        doc_agent = DocumentQAAgent(chat_tool, model_name=selected_model, callbacks=[debug_logger])
        summary_agent = SummaryAgent(summary_tool, model_name=selected_model, callbacks=[debug_logger])
        sql_agent = SQLAgent(sql_tool, model_name=selected_model, callbacks=[debug_logger])
        excel_agent = ExcelAgent(excel_tool, model_name=selected_model, callbacks=[debug_logger])
        web_agent = WebSearchAgent(web_tool, model_name=selected_model, callbacks=[debug_logger]) if (use_web_search and web_tool is not None) else None

        reasoning_agent = ReasoningAgent(model_name=selected_model, callbacks=[debug_logger])

        # Historial reciente
        recent_messages = (
            Message.query
            .filter_by(session_id=session.id, user_id=current_user.id)
            .order_by(Message.timestamp.desc())
            .limit(10)
            .all()
        )

        recent_messages.reverse()

        chat_history_for_router = [
            (HumanMessage(content=msg.content) if msg.sender == "user" else AIMessage(content=msg.content))
            for msg in recent_messages
        ]

        paired_history = []
        for i in range(0, len(recent_messages) - 1, 2):
            if recent_messages[i].sender == "user" and recent_messages[i + 1].sender == "bot":
                paired_history.append((recent_messages[i].content, recent_messages[i + 1].content))

        # ==================== OVERRIDE (SQL / AMBAS / DOCS por defecto) ====================
        user_mode, cleaned_question = _extract_user_mode(question_text)

        # 0) Prioridad: si el usuario pide "resumen", respetamos el tool de resumen
        qt = (cleaned_question or question_text or "").lower()
        wants_summary = any(k in qt for k in [
            "resumen", "resume", "summary", "síntesis", "resumen ejecutivo"
        ])

        if wants_summary:
            forced_results = []
            with get_openai_callback() as cb:
                # Intentar extraer un hint básico de documento, si existe
                doc_hint_local = ""
                m = re.search(r"(?:del|de la|del documento|documento|pdf|pptx?|presentación)\s+([a-z0-9_\-\. ]+)", qt)
                if m:
                    doc_hint_local = (m.group(1) or "").strip()

                r_sum = summary_tool.run({"doc_name_hint": doc_hint_local}, callbacks=[debug_logger])
                r_sum = r_sum if isinstance(r_sum, dict) else {"answer": str(r_sum), "source_documents": []}
                r_sum["origin"] = summary_tool.name
                forced_results.append(r_sum)

                project.cost += calculate_cost(selected_model, cb.prompt_tokens, cb.completion_tokens)

            final_result = reasoning_agent.run(cleaned_question, forced_results)

            # ✅ Devolución estándar (mismo flujo que el resto de overrides)
            answer_text = final_result.get("answer") or "Error generando respuesta."
            raw_sources = final_result.get("source_documents", [])

            sources_formatted = []
            for doc in raw_sources:
                if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
                    clean_meta = clean_metadata_for_json(doc.metadata)
                    sources_formatted.append({"page_content": doc.page_content, "metadata": clean_meta})
                elif isinstance(doc, dict):
                    sources_formatted.append(doc)

            user_msg = Message(session_id=session.id, user_id=current_user.id, sender="user", content=question_text)
            bot_msg = Message(session_id=session.id, user_id=current_user.id, sender="bot", content=answer_text, sources=sources_formatted)
            db.session.add(user_msg)
            db.session.add(bot_msg)

            # ✅ Auto-renombrar si aún es "Nuevo chat"
            if (session.name or "").strip().lower() in ("nuevo chat", "new chat"):
                session.name = _make_chat_title_from_question(cleaned_question or question_text)

            # ==================== APRENDIZAJE DE MEMORIA ====================
            if len(question_text) > 15:
                memory_store.add_fact(
                    text=f"Usuario preguntó: {question_text}",
                    metadata={
                        "type": "user_fact",
                        "session_id": session.id,
                    },
                )

            db.session.commit()

            return jsonify({
                "success": True,
                "answer": answer_text,
                "sources": sources_formatted,
                "user_message_id": user_msg.id,
                "bot_message_id": bot_msg.id,
            })

        elif user_mode in ("sql", "ambas") or user_mode is None:
            # Reglas de negocio solicitadas:
            # 1) "SQL"   -> solo SQL
            # 2) "AMBAS" -> SQL primero y luego DOCS para enriquecer
            # 3) (sin prefijo) -> DOCS directo
            forced_results = []

            with get_openai_callback() as cb:
                if user_mode == "sql":
                    r_sql = sql_agent.run(cleaned_question)
                    r_sql["origin"] = sql_tool.name
                    forced_results.append(r_sql)

                elif user_mode == "ambas":
                    # 1) SQL primero
                    r_sql = sql_agent.run(cleaned_question)
                    r_sql["origin"] = sql_tool.name
                    forced_results.append(r_sql)

                    # 2) DOCS después, enriqueciendo con lo obtenido de SQL
                    sql_raw = (r_sql.get("sql_raw_output") or r_sql.get("answer") or "").strip()
                    MAX_SQL_FOR_DOC_ENRICH = 6000
                    sql_snippet = sql_raw[:MAX_SQL_FOR_DOC_ENRICH]

                    enrich_question = (
                        f"{cleaned_question}\n\n"
                        "RESULTADO SQL (para enriquecer con documentos internos):\n"
                        f"{sql_snippet}"
                    )

                    # Memoria relevante
                    memory_docs = memory_store.recall(cleaned_question, k=5)

                    r_doc = doc_agent.run(
                        enrich_question,
                        paired_history,
                        extra_documents=memory_docs
                    )
                    r_doc["origin"] = chat_tool.name
                    forced_results.append(r_doc)

                else:
                    # (sin prefijo) -> DOCS directo
                    memory_docs = memory_store.recall(cleaned_question, k=5)
                    r_doc = doc_agent.run(
                        cleaned_question,
                        paired_history,
                        extra_documents=memory_docs
                    )
                    r_doc["origin"] = chat_tool.name
                    forced_results.append(r_doc)

                project.cost += calculate_cost(selected_model, cb.prompt_tokens, cb.completion_tokens)

            final_result = reasoning_agent.run(cleaned_question, forced_results)

            answer_text = final_result.get("answer") or "Error generando respuesta."
            raw_sources = final_result.get("source_documents", [])

            sources_formatted = []
            for doc in raw_sources:
                if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
                    clean_meta = clean_metadata_for_json(doc.metadata)
                    sources_formatted.append({"page_content": doc.page_content, "metadata": clean_meta})
                elif isinstance(doc, dict):
                    sources_formatted.append(doc)

            user_msg = Message(session_id=session.id, user_id=current_user.id, sender="user", content=question_text)
            bot_msg = Message(session_id=session.id, user_id=current_user.id, sender="bot", content=answer_text, sources=sources_formatted)
            db.session.add(user_msg)
            db.session.add(bot_msg)

            # ✅ Auto-renombrar si aún es "Nuevo chat"
            if (session.name or "").strip().lower() in ("nuevo chat", "new chat"):
                session.name = _make_chat_title_from_question(cleaned_question or question_text)

            # ==================== APRENDIZAJE DE MEMORIA ====================
            if len(question_text) > 15:
                memory_store.add_fact(
                    text=f"Usuario preguntó: {question_text}",
                    metadata={
                        "type": "user_fact",
                        "session_id": session.id,
                    },
                )

            db.session.commit()

            return jsonify({
                "success": True,
                "answer": answer_text,
                "sources": sources_formatted,
                "user_message_id": user_msg.id,
                "bot_message_id": bot_msg.id,
            })




        # ==================== ROUTER PRINCIPAL ====================
        router = AgentRouter(
            model_name=selected_model,
            tools=tools,
            doc_path=project.document_path,
        )

        # ✅ Pre-router determinista: si el usuario pide "documento X", forzamos DOCS
        qt = (question_text or "").lower()

        looks_like_doc_request = (
            "documento" in qt or "pdf" in qt or "ppt" in qt or "pptx" in qt or "presentación" in qt
        )

        # Heurística simple: última palabra con pinta de "nombre propio" o token raro (Honimunn)
        # (mejorable, pero ya evita el 80% de cagadas)
        import re
        doc_hint = None
        m = re.search(r"(documento|pdf|pptx?|presentación)\s+([A-Za-z0-9_\-\.]{3,})", question_text, flags=re.IGNORECASE)
        if m:
            doc_hint = m.group(2)

        qt = (question_text or "").lower()

        # 1) Detectar intención de resumen (prioridad alta)
        wants_summary = any(k in qt for k in [
            "resumen", "resume", "summary", "síntesis", "resumen ejecutivo"
        ])

        # 2) Intentar sacar hint de doc (si lo hay)
        #    - si ya tienes doc_hint, úsalo
        #    - si no, intenta extraer "del documento X"
        doc_hint_local = doc_hint
        if not doc_hint_local:
            m = re.search(r"(?:documento|pdf|ppt|presentación)\s+([a-z0-9_\- ]+)", qt)
            if m:
                doc_hint_local = (m.group(1) or "").strip()

        if wants_summary:
            # ✅ Forzamos RESUMEN (tool específico)
            class _ForcedChoice:
                def __init__(self, tool_name: str, doc_name_hint: str):
                    self.tool_calls = [{"name": tool_name, "args": {"doc_name_hint": doc_name_hint}}]
                    self.content = ""

            tool_choice = _ForcedChoice(summary_tool.name, (doc_hint_local or "").strip())

        elif looks_like_doc_request or doc_hint_local:
            # ✅ Forzamos DOCS (chat)
            class _ForcedChoice:
                def __init__(self, tool_name: str, question: str):
                    self.tool_calls = [{"name": tool_name, "args": {"question": question}}]
                    self.content = ""

            tool_choice = _ForcedChoice(chat_tool.name, question_text)

        else:
            tool_choice = router.route(
                question_text,
                chat_history_for_router,
                callbacks=[debug_logger],
            )


            # ✅ Guardarraíl: si el router dice DIRECTO pero no es smalltalk, forzamos DOCS
            raw_content = (getattr(tool_choice, "content", "") or "").strip()
            is_router_direct = raw_content.upper().startswith("ROUTE: DIRECTO")

            qt2 = (question_text or "").strip().lower()
            is_smalltalk = bool(re.match(r"^(hola|buenas|hey|gracias|ok|vale|perfecto|adios|hasta luego)\b", qt2)) or len(qt2.split()) <= 2

            if is_router_direct and not is_smalltalk:
                class _ForcedChoice:
                    def __init__(self, tool_name: str, question: str):
                        self.tool_calls = [{"name": tool_name, "args": {"question": question}}]
                        self.content = ""

                tool_choice = _ForcedChoice(chat_tool.name, question_text)



        # Si no hay tool_calls (respuesta directa)
        if not getattr(tool_choice, "tool_calls", None):
            raw_content = (getattr(tool_choice, "content", "") or "").strip()

            answer_text = raw_content
            if raw_content:
                first, *rest = raw_content.splitlines()
                if first.strip().upper().startswith("ROUTE:"):
                    answer_text = "\n".join(rest).strip()

            if not answer_text:
                answer_text = "No tengo respuesta para eso."

            sources_formatted = []  # ✅ FIX: en respuesta directa no hay fuentes

            user_msg = Message(
                session_id=session.id,
                user_id=current_user.id,
                sender="user",
                content=question_text
            )
            bot_msg = Message(
                session_id=session.id,
                user_id=current_user.id,
                sender="bot",
                content=answer_text,
                sources=sources_formatted
            )
            db.session.add(user_msg)
            db.session.add(bot_msg)

            if (session.name or "").strip().lower() in ("nuevo chat", "new chat"):
                session.name = _make_chat_title_from_question(question_text)


            # ==================== APRENDIZAJE DE MEMORIA ====================
            # Guardamos solo hechos con cierta entidad (no ruido)
            if len(question_text) > 15:
                memory_store.add_fact(
                    text=f"Usuario preguntó: {question_text}",
                    metadata={
                        "type": "user_fact",
                        "session_id": session.id,
                    },
                )


            db.session.commit()

            return jsonify({
                "success": True,
                "answer": answer_text,
                "sources": sources_formatted,
                "user_message_id": user_msg.id,
                "bot_message_id": bot_msg.id,
            })



        executed_results = []
        sql_context_doc = None  # ✅ para híbrido

        with get_openai_callback() as cb:
            for chosen_tool in (tool_choice.tool_calls or []):
                tool_name = chosen_tool.get("name")
                tool_args = chosen_tool.get("args", {}) or {}

                step_result = None

                if tool_name == chat_tool.name:
                    q_arg = tool_args.get("question", question_text)
                    memory_docs = memory_store.recall(q_arg, k=5)

                    extra_docs = list(memory_docs or [])
                    # ✅ Si venimos de SQL (híbrido), inyectamos un “doc” con el output SQL
                    if sql_context_doc is not None:
                        extra_docs = [sql_context_doc] + extra_docs

                    step_result = doc_agent.run(
                        q_arg,
                        paired_history,
                        extra_documents=extra_docs
                    )

                elif tool_name == summary_tool.name:
                    # si el router pasa hint, lo usamos
                    hint = tool_args.get("doc_name_hint", "") or ""
                    # OJO: si summary_tool es un BaseTool, usa summary_tool.run(...)
                    # Si es tu wrapper agent que expone .tool, deja .tool.run(...)
                    step_result = summary_tool.run({"doc_name_hint": hint}, callbacks=[debug_logger])

                elif tool_name == sql_tool.name:
                    q_arg = tool_args.get("query") or question_text
                    step_result = sql_agent.run(q_arg)

                    # ✅ Construimos contexto SQL compacto para el siguiente paso DOCS
                    if isinstance(step_result, dict):
                        sql_text = (step_result.get("sql_raw_output") or step_result.get("answer") or "").strip()
                    else:
                        sql_text = str(step_result or "").strip()

                    if sql_text:
                        compact = "\n".join([ln for ln in sql_text.splitlines() if ln.strip()][:25])
                        sql_context_doc = Document(
                            page_content=f"[SALIDA SQL - RESUMEN]\n{compact}",
                            metadata={"source": "SQL", "type": "sql_context"}
                        )

                elif web_agent is not None and tool_name == web_tool.name:
                    q_arg = tool_args.get("query") or question_text
                    step_result = web_agent.run(q_arg)

                else:
                    step_result = {"answer": f"No sé qué herramienta usar para: {tool_name}", "source_documents": []}

                # ✅ Normalizamos resultado para que siempre sea dict
                if step_result is None:
                    step_result = {"answer": "Error interno en la herramienta.", "source_documents": []}
                elif not isinstance(step_result, dict):
                    step_result = {"answer": str(step_result), "source_documents": []}

                step_result["origin"] = tool_name
                executed_results.append(step_result)

        # ✅ El callback acumula tokens de TODO lo ejecutado dentro del with
        project.cost += calculate_cost(selected_model, cb.prompt_tokens, cb.completion_tokens)



        final_result = reasoning_agent.run(question_text, executed_results)

        answer_text = final_result.get("answer") or "Error generando respuesta."
        raw_sources = final_result.get("source_documents", [])

        # UI: Preview SQL si aplica
        sql_result = next((r for r in executed_results if r.get("origin") == sql_tool.name), None)
        sql_raw = (sql_result.get("sql_raw_output") or "").strip() if sql_result else ""
        if sql_raw:
            lines = [ln for ln in sql_raw.splitlines() if ln.strip()]
            header = lines[0] if lines else "Resultados:"
            rows = lines[1:] if len(lines) > 1 else []
            preview_n = 10
            preview = "\n".join([header] + rows[:preview_n]).strip()

            answer_text += (
                "\n\n---\n\n"
                "### Preview de datos (primeras 10 filas)\n"
                "```text\n"
                f"{preview}\n"
                "```\n\n"
                "<details>\n"
                "<summary><b>Ver tabla completa</b></summary>\n\n"
                "```text\n"
                f"{sql_raw}\n"
                "```\n"
                "</details>\n"
            )

        sources_formatted = []
        for doc in raw_sources:
            if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
                clean_meta = clean_metadata_for_json(doc.metadata)
                sources_formatted.append({"page_content": doc.page_content, "metadata": clean_meta})
            elif isinstance(doc, dict):
                sources_formatted.append(doc)

        user_msg = Message(session_id=session.id, user_id=current_user.id, sender="user", content=question_text)
        bot_msg = Message(session_id=session.id, user_id=current_user.id, sender="bot", content=answer_text, sources=sources_formatted)
        db.session.add(user_msg)
        db.session.add(bot_msg)

        # ✅ Auto-renombrar si aún es "Nuevo chat"
        if (session.name or "").strip().lower() in ("nuevo chat", "new chat"):
            session.name = _make_chat_title_from_question(cleaned_question or question_text)

        # ==================== APRENDIZAJE DE MEMORIA ====================
        # Guardamos solo hechos con cierta entidad (no ruido)
        if len(question_text) > 15:
            memory_store.add_fact(
                text=f"Usuario preguntó: {question_text}",
                metadata={
                    "type": "user_fact",
                    "session_id": session.id,
                },
            )


        db.session.commit()


        print("✅ RESPUESTA GENERADA Y GUARDADA.\n" + "=" * 60 + "\n")

        return jsonify({
            "success": True,
            "answer": answer_text,
            "sources": sources_formatted,
            "user_message_id": user_msg.id,
            "bot_message_id": bot_msg.id,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error del servidor: {str(e)}"}), 500


# ==================== EDICIÓN Y LIMPIEZA ====================

@bp.route("/clear_history/<session_id>", methods=["POST"])
@login_required
def clear_history(session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()

    try:
        session.messages.delete()
        db.session.commit()
        _clear_chain_cache_for_project(session.project_id)
        flash("Historial borrado.", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Error: {e}", "danger")

    return redirect(url_for("main.chat_session", session_id=session_id))


@bp.route("/edit_and_resubmit/<int:message_id>", methods=["POST"])
@login_required
def edit_and_resubmit(message_id):
    user_message = Message.query.filter_by(id=message_id, user_id=current_user.id).first_or_404()

    session = user_message.session
    project = session.project
    new_text = (request.json or {}).get("new_question")
    model_for_regen = project.model_name or "gpt-4o"

    if not new_text or user_message.sender != "user":
        return jsonify({"error": "Inválido"}), 400

    debug_logger = ConsoleLogger()

    try:
        # 1) Borrar mensajes posteriores SOLO del usuario actual (blindaje)
        msgs_to_del = Message.query.filter(
            Message.session_id == session.id,
            Message.user_id == current_user.id,
            Message.timestamp > user_message.timestamp,
        ).all()
        for m in msgs_to_del:
            db.session.delete(m)

        # 2) Actualizar contenido del mensaje de usuario
        user_message.content = new_text

        # 3) Reconstruir histórico previo (para router / docs)
        prev_msgs = (
            Message.query.filter(
                Message.session_id == session.id,
                Message.user_id == current_user.id,
                Message.timestamp < user_message.timestamp,
            )
            .order_by(Message.timestamp.asc())
            .all()
        )

        chat_history_router = [
            (HumanMessage(content=m.content) if m.sender == "user" else AIMessage(content=m.content))
            for m in prev_msgs
        ]

        paired_history = []
        for i in range(0, len(prev_msgs) - 1, 2):
            if prev_msgs[i].sender == "user" and prev_msgs[i + 1].sender == "bot":
                paired_history.append((prev_msgs[i].content, prev_msgs[i + 1].content))

        # 4) Herramientas
        chat_tool = ChatWithDocumentTool(
            project_id=project.id,
            vector_store_path=project.vector_store_path,
            model_name=model_for_regen,
            project_settings=project.settings or {},
        )
        summary_tool = SummarizeDocumentTool(
            project_id=project.id,
            vector_store_path=project.vector_store_path,
            model_name=model_for_regen,
        )
        sql_tool = SQLDatabaseTool(
            model_name=model_for_regen,
            project_settings=project.settings or {},
        )
        excel_tool = ExcelAnalysisTool(
            doc_path=project.document_path,
            model_name=model_for_regen,
        )

        tools = [chat_tool, summary_tool, sql_tool, excel_tool]


        # Agentes
        doc_agent = DocumentQAAgent(chat_tool, model_name=model_for_regen, callbacks=[debug_logger])
        summary_agent = SummaryAgent(summary_tool, model_name=model_for_regen, callbacks=[debug_logger])
        sql_agent = SQLAgent(sql_tool, model_name=model_for_regen, callbacks=[debug_logger])
        excel_agent = ExcelAgent(excel_tool, model_name=model_for_regen, callbacks=[debug_logger])
        reasoning_agent = ReasoningAgent(model_name=model_for_regen, callbacks=[debug_logger])

        # Override
        user_mode, cleaned_question = _extract_user_mode(new_text)

        if user_mode in ("sql", "doc", "hib"):
            forced_results = []

            with get_openai_callback() as cb:
                if user_mode == "sql":
                    r_sql = sql_agent.run(cleaned_question)
                    r_sql["origin"] = sql_tool.name
                    forced_results.append(r_sql)

                elif user_mode == "doc":
                    # 1️⃣ Recuperar memoria relevante
                    memory_docs = memory_store.recall(cleaned_question, k=5)

                    # 2️⃣ Inyectar memoria como contexto adicional
                    r_doc = doc_agent.run(
                        cleaned_question,
                        paired_history,
                        extra_documents=memory_docs
                    )

                    r_doc["origin"] = chat_tool.name
                    forced_results.append(r_doc)

                elif user_mode == "hib":
                    r_sql = sql_agent.run(cleaned_question)
                    r_sql["origin"] = sql_tool.name
                    forced_results.append(r_sql)

                    # 1️⃣ Recuperar memoria relevante
                    memory_docs = memory_store.recall(cleaned_question, k=5)

                    # 2️⃣ Inyectar memoria como contexto adicional
                    r_doc = doc_agent.run(
                        cleaned_question,
                        paired_history,
                        extra_documents=memory_docs
                    )

                    r_doc["origin"] = chat_tool.name
                    forced_results.append(r_doc)

                project.cost += calculate_cost(model_for_regen, cb.prompt_tokens, cb.completion_tokens)

            final_result = reasoning_agent.run(cleaned_question, forced_results)

            ans = final_result.get("answer") or "Error generando respuesta."
            raw_sources = final_result.get("source_documents", [])

            srcs = []
            for d in raw_sources:
                if hasattr(d, "page_content") and hasattr(d, "metadata"):
                    srcs.append({"page_content": d.page_content, "metadata": clean_metadata_for_json(d.metadata)})
                elif isinstance(d, dict):
                    srcs.append(d)

            bot_msg = Message(session_id=session.id, user_id=current_user.id, sender="bot", content=ans, sources=srcs)
            db.session.add(bot_msg)
            db.session.commit()

            return jsonify({
                "success": True,
                "answer": ans,
                "sources": srcs,
                "user_message_id": user_message.id,
                "bot_message_id": bot_msg.id,
            })

        # Router normal (auto)
        router = AgentRouter(
            model_name=model_for_regen,
            tools=tools,
            doc_path=project.document_path,
        )

        choice = router.route(
            new_text,
            chat_history_router,
            callbacks=[debug_logger],
        )

        executed_results = []

        with get_openai_callback() as cb:
            for chosen_tool in (choice.tool_calls or []):
                tool_name = chosen_tool.get("name")
                tool_args = chosen_tool.get("args", {}) or {}

                step_result = None

                if tool_name == chat_tool.name:
                    q_arg = tool_args.get("question", new_text)
                    memory_docs = memory_store.recall(q_arg, k=5)

                    step_result = doc_agent.run(
                        q_arg,
                        paired_history,
                        extra_documents=memory_docs
                    )

                elif tool_name == summary_tool.name:
                    hint = tool_args.get("doc_name_hint", "") or ""
                    step_result = summary_tool.run({"doc_name_hint": hint}, callbacks=[debug_logger])

                elif tool_name == sql_tool.name:
                    q_arg = tool_args.get("query") or new_text
                    step_result = sql_agent.run(q_arg)

                elif tool_name == excel_tool.name:
                    q_arg = tool_args.get("query") or tool_args.get("question") or new_text
                    file_hint = tool_args.get("file_name_hint", "")
                    step_result = excel_agent.run(q_arg, file_hint)

                else:
                    step_result = doc_agent.run(new_text, paired_history)

                if step_result is None:
                    step_result = {"answer": "Error interno.", "source_documents": []}

                step_result["origin"] = tool_name
                executed_results.append(step_result)

            project.cost += calculate_cost(model_for_regen, cb.prompt_tokens, cb.completion_tokens)

        if not executed_results:
            executed_results.append({
                "answer": "No se ejecutó ninguna herramienta. Fallback a documentos.",
                "source_documents": [],
                "origin": chat_tool.name,
            })

        final_result = reasoning_agent.run(new_text, executed_results)

        ans = final_result.get("answer") or "Error generando respuesta."
        raw_sources = final_result.get("source_documents", [])

        srcs = []
        for d in raw_sources:
            if hasattr(d, "page_content") and hasattr(d, "metadata"):
                srcs.append({"page_content": d.page_content, "metadata": clean_metadata_for_json(d.metadata)})
            elif isinstance(d, dict):
                srcs.append(d)

        bot_msg = Message(session_id=session.id, user_id=current_user.id, sender="bot", content=ans, sources=srcs)
        db.session.add(bot_msg)
        db.session.commit()

        return jsonify({
            "success": True,
            "answer": ans,
            "sources": srcs,
            "user_message_id": user_message.id,
            "bot_message_id": bot_msg.id,
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error regen: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/check_status")
def check_status():
    projects = Project.query.all()
    statuses = {p.id: p.status for p in projects}
    return jsonify(statuses)


@bp.route("/settings/<project_id>", methods=["GET", "POST"])
@login_required
def project_settings(project_id):
    project = Project.query.get_or_404(project_id)

    if request.method == "POST":
        current_settings = project.settings or {}
        current_settings["k_value"] = int(request.form.get("k_value", 5))
        tpl = request.form.get("template_selected", "")

        if tpl and tpl != "custom":
            current_settings["system_instruction"] = get_template_prompt(tpl)
            current_settings["template_type"] = tpl
        else:
            custom = (request.form.get("system_instruction", "") or "").strip()
            current_settings["system_instruction"] = custom
            current_settings["template_type"] = "custom" if custom else "generico"

        current_settings["sql_context"] = request.form.get("sql_context", "")
        project.settings = current_settings
        db.session.commit()

        _clear_chain_cache_for_project(project.id)


        flash("Ajustes guardados.", "success")
        return redirect(url_for("main.project_settings", project_id=project.id))

    settings = project.settings or {}
    return render_template(
        "settings.html",
        project=project,
        settings=settings,
        templates=get_all_templates(),
        current_template=settings.get("template_type", "generico"),
        current_instruction=settings.get("system_instruction", ""),
    )


# ==================== AUTH ====================

@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        
        user = User.query.filter_by(username=username, is_active=True).first()
        if user and user.check_password(password):
            login_user(user)

            # Auditoría básica (sin IP)
            user.last_login = datetime.utcnow()

            # Cerrar sesión anterior abierta si existiera (por seguridad/consistencia)
            prev = (
                LoginSession.query
                .filter_by(user_id=user.id, ended_at=None)
                .order_by(LoginSession.started_at.desc())
                .first()
            )
            if prev:
                now = datetime.utcnow()
                prev.ended_at = now
                prev.duration_sec = max(0, int((now - prev.started_at).total_seconds()))

            # Crear nueva sesión de login
            ls = LoginSession(
                user_id=user.id,
                started_at=datetime.utcnow(),
                n_questions=0,
                last_activity_at=datetime.utcnow(),
            )
            db.session.add(ls)
            db.session.commit()

            flask_session["login_session_id"] = ls.id

            return redirect(url_for("main.index"))



        flash("Credenciales incorrectas", "danger")

    return render_template("login.html")


@bp.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    try:
        ls_id = flask_session.get("login_session_id")
        if ls_id:
            ls = LoginSession.query.filter_by(id=ls_id, user_id=current_user.id).first()
            if ls and ls.ended_at is None:
                now = datetime.utcnow()
                ls.ended_at = now
                ls.duration_sec = max(0, int((now - ls.started_at).total_seconds()))
                db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass

    flask_session.pop("login_session_id", None)

    logout_user()
    return redirect(url_for("main.login"))


@bp.route("/admin/activity")
@login_required
def admin_activity():
    if getattr(current_user, "role", None) != "admin":
        abort(403)

    # Redirigir al shell único (tab activity)
    project = _get_or_create_single_project()

    # llevarte a una sesión válida para que el sidebar no rompa
    chat_sessions_list = (
        ChatSession.query
        .filter_by(project_id=project.id, user_id=current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    current_chat_session = chat_sessions_list[0] if chat_sessions_list else None

    if current_chat_session:
        return redirect(url_for("main.index", tab="activity", session=current_chat_session.id))

    # Si por lo que sea no hay sesiones, index ya crea una al entrar,
    # pero aquí le mandamos sin session
    return redirect(url_for("main.index", tab="activity"))





@bp.route("/admin/activity/export_sessions.csv")
@login_required
def export_sessions_csv():
    if getattr(current_user, "role", None) != "admin":
        abort(403)

    import csv
    import io

    def row(v):
        # Evita None y deja el CSV limpio
        return "" if v is None else v

    def dt(v):
        # Formato consistente y “humano” (sin milisegundos)
        return "" if v is None else v.strftime("%Y-%m-%d %H:%M:%S")

    @stream_with_context
    def generate():
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        # Cabecera
        writer.writerow([
            "login_session_id",
            "user_id",
            "username",
            "started_at",
            "last_activity_at",
            "ended_at",
            "duration_sec",
            "n_questions",
        ])
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)

        # Query histórico completo (streaming)
        q = (
            db.session.query(LoginSession, User)
            .join(User, User.id == LoginSession.user_id)
            .order_by(LoginSession.started_at.desc())
            .yield_per(1000)
        )

        for ls, u in q:
            writer.writerow([
                row(ls.id),
                row(u.id),
                row(u.username),
                dt(ls.started_at),
                dt(ls.last_activity_at),
                dt(ls.ended_at),
                row(ls.duration_sec),
                row(ls.n_questions),
            ])
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    filename = "detalles_sesiones_historico.csv"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return Response(generate(), mimetype="text/csv", headers=headers)


