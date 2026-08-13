"""
test_pipeline.py — what sits between the endpoint and the tools: the
request-level guardrails, tool dispatch, the final formatting agent and the
BM25 index.

The logic of the guardrails is already tested in test_guardrails.py. What is
tested here is what actually protects: that they CUT, and that they cut BEFORE
the model is called and BEFORE anything is written to the database.
"""

import os
import pickle
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from app.main.guards import _dlp_block, _quota_block
from app.main.pipeline import (
    _calls_from_override,
    _run_tools,
    _sql_context_document,
)
from app.models import Message
from app.rag_logic.agent_reasoning import (
    ReasoningAgent,
    _build_contributions_summary,
    _merge_source_docs,
)
from app.rag_logic.bm25_index import (
    build_bm25_retriever,
    load_bm25_index,
    persist_bm25_index,
)


# ══════════════════════════════════════════════════════════════════════════
# Guardrails: that they cut, and that they cut in time
# ══════════════════════════════════════════════════════════════════════════

class TestDailyQuota:

    def test_without_variable_not_there_is_quota(self, app, monkeypatch):
        """Railway does not need it: OpenAI is prepaid and stops itself."""
        monkeypatch.delenv("DAILY_QUESTION_LIMIT", raising=False)
        with app.test_request_context():
            assert _quota_block() is None

    def test_zero_desactiva_the_quota(self, app, monkeypatch):
        monkeypatch.setenv("DAILY_QUESTION_LIMIT", "0")
        with app.test_request_context():
            assert _quota_block() is None

    def test_by_below_of_the_cap_lets_pasar(self, app, db, test_user, test_chat_session, monkeypatch):
        monkeypatch.setenv("DAILY_QUESTION_LIMIT", "5")
        db.session.add(Message(session_id=test_chat_session.id, user_id=test_user.id,
                               sender="user", content="una"))
        db.session.commit()
        with app.test_request_context():
            with patch("app.main.guards.current_user", test_user):
                assert _quota_block() is None

    def test_at_reaching_the_cap_returns_429(self, app, db, test_user, test_chat_session, monkeypatch):
        monkeypatch.setenv("DAILY_QUESTION_LIMIT", "2")
        for i in range(2):
            db.session.add(Message(session_id=test_chat_session.id, user_id=test_user.id,
                                   sender="user", content=f"p{i}"))
        db.session.commit()
        with app.test_request_context():
            with patch("app.main.guards.current_user", test_user):
                blocked = _quota_block()
        assert blocked is not None
        body, status = blocked
        assert status == 429
        assert body.get_json()["quota_limit"] == 2

    def test_only_count_the_messages_of_the_user(self, app, db, test_user, test_chat_session, monkeypatch):
        """Bot answers do not consume quota."""
        monkeypatch.setenv("DAILY_QUESTION_LIMIT", "2")
        for s in ("user", "bot", "bot"):
            db.session.add(Message(session_id=test_chat_session.id, user_id=test_user.id,
                                   sender=s, content="x"))
        db.session.commit()
        with app.test_request_context():
            with patch("app.main.guards.current_user", test_user):
                assert _quota_block() is None


class TestDLPAtTheEndpoint:

    def test_text_clean_passes(self, app):
        with app.test_request_context():
            assert _dlp_block("¿cuántos días de vacaciones tengo?", "ctx") is None

    def test_a_iban_returns_400(self, app):
        with app.test_request_context():
            blocked = _dlp_block("mi cuenta es ES9121000418450200051332", "ctx")
        assert blocked is not None
        body, status = blocked
        assert status == 400

    def test_the_message_of_error_not_repeats_the_value(self, app):
        """Returning the detected value in the response would leak it anyway."""
        iban = "ES9121000418450200051332"
        with app.test_request_context():
            body, _ = _dlp_block(f"cuenta {iban}", "ctx")
        assert iban not in body.get_json()["error"]


class TestGuardrailOrdering:

    def test_the_endpoint_cuts_before_of_construir_nothing(self, auth_client, test_chat_session, monkeypatch):
        """
        The property that matters: with PII in the question, neither the tools
        nor the router are ever constructed. Once the data reaches the model or
        the database, it has left the perimeter.
        """
        monkeypatch.delenv("DAILY_QUESTION_LIMIT", raising=False)
        with patch("app.main.pipeline._ToolBox") as box, \
             patch("app.main.pipeline.AgentRouter") as router:
            r = auth_client.post(
                f"/ask/{test_chat_session.id}",
                json={"question": "mi IBAN es ES9121000418450200051332"},
            )
        assert r.status_code == 400
        box.assert_not_called()
        router.assert_not_called()

    def test_a_question_bloqueada_not_persists(self, auth_client, db, test_chat_session, monkeypatch):
        monkeypatch.delenv("DAILY_QUESTION_LIMIT", raising=False)
        before = Message.query.filter_by(session_id=test_chat_session.id).count()
        auth_client.post(
            f"/ask/{test_chat_session.id}",
            json={"question": "mi IBAN es ES9121000418450200051332"},
        )
        assert Message.query.filter_by(session_id=test_chat_session.id).count() == before


# ══════════════════════════════════════════════════════════════════════════
# Overrides and tool dispatch
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def box():
    b = MagicMock()
    b.docs.name = "chat_with_documents"
    b.summary.name = "summarise_document"
    b.sql.name = "query_hr_database"
    b.excel.name = "excel_analyst"
    b.web = None
    return b


class TestOverrides:

    def test_no_prefix_lets_the_router_decide(self, box):
        calls, _ = _calls_from_override("cuál es la política de teletrabajo", box)
        assert calls is None

    def test_the_sql_prefix_produces_the_same_shape_as_the_router(self, box):
        """
        An override is not a different path: it skips the decision. It produces
        the same tool_calls list, so what runs afterwards is identical.
        """
        calls, cleaned = _calls_from_override("SQL: cuántos empleados hay", box)
        assert calls == [{"name": "query_hr_database", "args": {"query": "cuántos empleados hay"}}]
        assert cleaned == "cuántos empleados hay"

    def test_ambas_chains_sql_and_documents(self, box):
        calls, _ = _calls_from_override("AMBAS - salarios y política", box)
        assert [c["name"] for c in calls] == ["query_hr_database", "chat_with_documents"]

    def test_asking_for_a_summary_goes_to_the_summary_tool(self, box):
        calls, _ = _calls_from_override("dame un resumen del handbook", box)
        assert calls[0]["name"] == "summarise_document"


class TestToolDispatch:

    def test_an_unknown_tool_does_not_blow_up(self, box):
        r = _run_tools([{"name": "no_existe", "args": {}}], box, "q", [])
        assert "No tool available for" in r[0]["answer"]
        assert r[0]["source_documents"] == []

    def test_the_result_always_is_a_dict(self, box):
        """Tools may return a str; everything downstream assumes a dict."""
        box.summary.run.return_value = "una cadena suelta"
        r = _run_tools([{"name": "summarise_document", "args": {}}], box, "q", [])
        assert isinstance(r[0], dict)
        assert r[0]["answer"] == "una cadena suelta"

    def test_tags_the_origin_of_each_step(self, box):
        box.summary.run.return_value = {"answer": "x", "source_documents": []}
        r = _run_tools([{"name": "summarise_document", "args": {}}], box, "q", [])
        assert r[0]["origin"] == "summarise_document"

    def test_the_sql_result_is_chained_into_the_document_query(self, box):
        """Hybrid mode: the SQL output is chained in as context for DOCS."""
        box.sql_agent.run.return_value = {"answer": "t", "sql_raw_output": "dept | n\nEng | 12",
                                           "source_documents": []}
        box.docs.run.return_value = {"answer": "según la política...", "source_documents": []}
        _run_tools([{"name": "query_hr_database", "args": {"query": "q"}},
                    {"name": "chat_with_documents", "args": {"question": "q"}}], box, "q", [])
        question = box.docs.run.call_args[0][0]["question"]
        assert "SQL OUTPUT" in question

    def test_no_calls_returns_an_empty_list(self, box):
        assert _run_tools([], box, "q", []) == []
        assert _run_tools(None, box, "q", []) == []


class TestSQLContext:

    def test_prefiere_the_salida_raw(self):
        doc = _sql_context_document({"sql_raw_output": "TABLA", "answer": "prosa"})
        assert "TABLA" in doc.page_content

    def test_recorta_the_salidas_largas(self):
        doc = _sql_context_document({"sql_raw_output": "\n".join(f"row {i}" for i in range(200))})
        assert len(doc.page_content.splitlines()) < 40

    def test_without_contenido_not_produces_document(self):
        assert _sql_context_document({"sql_raw_output": "", "answer": ""}) is None


# ══════════════════════════════════════════════════════════════════════════
# Agente de formato final
# ══════════════════════════════════════════════════════════════════════════

class TestFormattingAgent:

    def test_prefiere_the_salida_raw_of_sql(self):
        """
        This is why a rewriting pass in SQLAgent would be dead code: its
        `answer` is never the one read.
        """
        text = _build_contributions_summary([
            {"origin": "query_hr_database", "sql_raw_output": "TABLA CRUDA", "answer": "REFORMULADO"},
        ])
        assert "TABLA CRUDA" in text
        assert "REFORMULADO" not in text

    def test_usa_answer_when_not_there_is_salida_sql(self):
        text = _build_contributions_summary([{"origin": "chat_with_documents", "answer": "la response"}])
        assert "la response" in text

    def test_trunca_the_bloques_enormes(self):
        text = _build_contributions_summary([{"origin": "x", "answer": "y" * 20000}])
        assert "TRUNCATED" in text

    def test_without_contribuciones_it_says(self):
        assert "No useful response" in _build_contributions_summary([])

    def test_fusiona_the_sources_of_all_the_pasos(self):
        docs = _merge_source_docs([
            {"source_documents": [Document(page_content="a")]},
            {"source_documents": [Document(page_content="b")]},
            {"source_documents": []},
        ])
        assert [d.page_content for d in docs] == ["a", "b"]

    def test_returns_answer_and_sources(self):
        with patch("app.rag_logic.agent_reasoning.get_llm") as get_llm:
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="  final answer  ")
            get_llm.return_value = llm
            r = ReasoningAgent(model_name="gpt-4o-mini").run(
                "question", [{"origin": "x", "answer": "dato", "source_documents": [Document(page_content="f")]}],
            )
        assert r["answer"] == "final answer"
        assert len(r["source_documents"]) == 1


# ══════════════════════════════════════════════════════════════════════════
# Persisted BM25 index
# ══════════════════════════════════════════════════════════════════════════

class TestBM25:

    def test_builds_from_texts_and_metadata(self):
        r = build_bm25_retriever(["text uno", "text dos"], [{"a": 1}, {"a": 2}])
        assert r is not None

    def test_without_documents_returns_none(self):
        assert build_bm25_retriever([], []) is None

    def test_longitudes_desiguales_return_none(self):
        """A silent zip() would drop documents without warning."""
        assert build_bm25_retriever(["a", "b"], [{"x": 1}]) is None

    def test_ida_and_vuelta_by_disco(self, tmp_path):
        vs = MagicMock()
        vs.get.return_value = {"documents": ["política de vacaciones", "guía de onboarding"],
                               "metadatas": [{"f": "a.pdf"}, {"f": "b.pdf"}]}
        assert persist_bm25_index(vs, str(tmp_path)) is True
        assert load_bm25_index(str(tmp_path)) is not None

    def test_without_file_returns_none(self, tmp_path):
        assert load_bm25_index(str(tmp_path)) is None

    def test_a_pickle_corrupto_not_brings_down_the_query(self, tmp_path):
        """
        An old pickle may fail to load after a library upgrade. The query must
        keep working without the lexical leg rather than blow up.
        """
        (tmp_path / "_bm25_index.pkl").write_bytes(b"esto no es un pickle")
        assert load_bm25_index(str(tmp_path)) is None

    def test_a_vector_store_empty_not_persists_nothing(self, tmp_path):
        vs = MagicMock()
        vs.get.return_value = {"documents": [], "metadatas": []}
        assert persist_bm25_index(vs, str(tmp_path)) is False
