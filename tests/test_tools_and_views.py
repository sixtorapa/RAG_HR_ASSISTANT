"""
test_tools_and_views.py — the Excel department guardrail, the SQL agent, the
console logger, web search, the summariser and the screens.

The first block is the one that matters. `ExcelAnalysisTool._find_excel_files`
is an ACCESS control, not a convenience: without it the tool could read any
.xlsx in the corpus, bypassing the department filter that document retrieval
does apply. A guardrail that exists on only one of two paths is not a guardrail.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from app.rag_logic.agent_sql import (
    SQLAgent,
    _build_context_from_docs,
    _normalize_result,
)
from app.rag_logic.console_logger import ConsoleLogger
from app.rag_logic.excel_tool import ExcelAnalysisTool
from app.rag_logic.summarizer import _decorate_docs_with_source, _downsample_evenly, _meta_text
from app.rag_logic.web_search import WebSearchTool


@pytest.fixture
def corpus(tmp_path):
    """A knowledge_base with one .xlsx in three different departments."""
    for dep in ("compensation_benefits", "recruitment_talent", "it_workplace_policies"):
        d = tmp_path / dep
        d.mkdir()
        (d / f"{dep}_datos.xlsx").write_bytes(b"fake xlsx")
    (tmp_path / "raiz.xlsx").write_bytes(b"fake xlsx")
    return tmp_path


# ══════════════════════════════════════════════════════════════════════════
# Access guardrail for the Excel analysis tool
# ══════════════════════════════════════════════════════════════════════════

class TestExcelFileAccess:

    def _tool(self, corpus, allowed):
        return ExcelAnalysisTool(
            doc_path=str(corpus), model_name="gpt-4o-mini", allowed_departments=allowed,
        )

    def test_none_sees_everything_admin_only(self, corpus):
        # None is produced by User.get_allowed_departments() only when role=admin.
        files = self._tool(corpus, None)._find_excel_files()
        assert len(files) == 4

    def test_a_list_restricts_to_those_departments(self, corpus):
        files = self._tool(corpus, ["compensation_benefits"])._find_excel_files()
        assert len(files) == 1
        assert "compensation_benefits" in files[0]

    def test_not_cuela_the_excel_of_other_department(self, corpus):
        """The property that matters: what is forbidden does not appear."""
        files = self._tool(corpus, ["recruitment_talent"])._find_excel_files()
        assert not any("compensation_benefits" in f for f in files)

    def test_an_empty_list_sees_no_file(self, corpus):
        """
        Fail-closed, like the document RBAC: with no departments assigned you
        see nothing, rather than everything.
        """
        assert self._tool(corpus, [])._find_excel_files() == []

    def test_several_departments_permitidos(self, corpus):
        files = self._tool(corpus, ["compensation_benefits", "recruitment_talent"])._find_excel_files()
        assert len(files) == 2

    def test_ignoran_the_temporary_of_excel(self, corpus):
        (corpus / "compensation_benefits" / "~$borrador.xlsx").write_bytes(b"tmp")
        files = self._tool(corpus, ["compensation_benefits"])._find_excel_files()
        assert not any("~$" in f for f in files)

    def test_a_carpeta_missing_not_blows_up(self, tmp_path):
        tool = ExcelAnalysisTool(doc_path=str(tmp_path / "no_existe"),
                                 model_name="gpt-4o-mini", allowed_departments=None)
        assert tool._find_excel_files() == []

    def test_without_files_it_says_in_instead_of_failing(self, tmp_path):
        tool = ExcelAnalysisTool(doc_path=str(tmp_path), model_name="gpt-4o-mini",
                                 allowed_departments=None)
        assert "No Excel files were found" in tool._run("cualquier question")


class TestDataFrameCleanup:

    def test_strips_columns_and_rows_empty(self):
        import pandas as pd
        tool = ExcelAnalysisTool(doc_path="/x", model_name="m")
        df = pd.DataFrame({"a": [1, None], "vacia": [None, None]})
        cleaned = tool._preprocess_dataframe(df)
        assert "vacia" not in cleaned.columns

    def test_the_columns_unnamed_vacian(self):
        import pandas as pd
        tool = ExcelAnalysisTool(doc_path="/x", model_name="m")
        df = pd.DataFrame({"Unnamed: 0": [1, 2], "real": [3, 4]})
        assert "Unnamed: 0" not in tool._preprocess_dataframe(df).columns


# ══════════════════════════════════════════════════════════════════════════
# SQL agent (no LLM call)
# ══════════════════════════════════════════════════════════════════════════

class TestSQLAgent:

    def test_not_builds_cliente_llm(self):
        """
        A rewriting pass here would be discarded by the formatting agent, so
        building the client at all would be paying for an object per request
        for nothing.
        """
        tool = MagicMock()
        tool.name = "query_hr_database"
        with patch("app.rag_logic.agent_sql.get_llm") as get_llm:
            agent = SQLAgent(tool, model_name="gpt-4o")
            get_llm.assert_not_called()
        assert not hasattr(agent, "llm")

    def test_keeps_the_salida_raw(self):
        tool = MagicMock()
        tool.name = "query_hr_database"
        tool.run.return_value = {"answer": "TABLA + interpretación", "source_documents": []}
        r = SQLAgent(tool, model_name="gpt-4o").run("q")
        assert r["sql_raw_output"] == "TABLA + interpretación"
        assert r["answer"] == r["sql_raw_output"]

    def test_normalises_a_answer_in_text(self):
        assert _normalize_result("solo text") == {"answer": "solo text", "source_documents": []}

    def test_completes_the_keys_that_falten(self):
        r = _normalize_result({"answer": "x"})
        assert r["source_documents"] == []

    def test_the_contexto_includes_the_fuente(self):
        docs = [Document(page_content="contenido", metadata={"source": "politica.pdf"})]
        assert "politica.pdf" in _build_context_from_docs(docs)

    def test_the_contexto_accepts_dicts_serialized(self):
        docs = [{"page_content": "contenido", "metadata": {"source": "a.pdf"}}]
        assert "contenido" in _build_context_from_docs(docs)

    def test_without_documents_returns_string_empty(self):
        assert _build_context_from_docs([]) == ""


# ══════════════════════════════════════════════════════════════════════════
# Resumidor
# ══════════════════════════════════════════════════════════════════════════

class TestSummariser:

    def test_the_sampling_keeps_the_ends(self):
        """
        When trimming to fit the context, keeping the first N would lose the end
        of the document. Even sampling spreads the selection.
        """
        docs = [Document(page_content=f"d{i}") for i in range(100)]
        m = _downsample_evenly(docs, 10)
        assert len(m) == 10
        assert m[0].page_content == "d0"

    def test_when_they_all_fit_nothing_is_touched(self):
        docs = [Document(page_content=f"d{i}") for i in range(5)]
        assert _downsample_evenly(docs, 10) == docs

    def test_decorates_with_the_provenance(self):
        docs = [Document(page_content="texto", metadata={"filename": "a.pdf", "page_number": 2})]
        assert "a.pdf" in _decorate_docs_with_source(docs)[0].page_content

    def test_metadata_empty_not_blows_up(self):
        assert isinstance(_meta_text({}), str)


# ══════════════════════════════════════════════════════════════════════════
# Web search
# ══════════════════════════════════════════════════════════════════════════

class TestWebSearch:

    def test_returns_the_formato_comun(self):
        with patch("app.rag_logic.web_search.DuckDuckGoSearchRun") as ddg:
            ddg.return_value.invoke.return_value = "resultado de la búsqueda"
            r = WebSearchTool()._run("capital de Francia")
        assert r == {"answer": "resultado de la búsqueda", "source_documents": []}

    def test_a_fallo_of_red_not_brings_down_the_query(self):
        """
        It is the only tool that depends on a third party outside our
        control: tiene que degradar, no propagar.
        """
        with patch("app.rag_logic.web_search.DuckDuckGoSearchRun") as ddg:
            ddg.return_value.invoke.side_effect = RuntimeError("sin red")
            r = WebSearchTool()._run("lo que sea")
        assert "Error searching the web" in r["answer"]
        assert r["source_documents"] == []


# ══════════════════════════════════════════════════════════════════════════
# Logger de consola
# ══════════════════════════════════════════════════════════════════════════

class TestConsoleLogger:
    """
    It is a callback, passed to EVERY call in the system. A failure in it
    tumbaría la petición entera, así que lo que hay que probar es que no puede.
    """

    def test_a_ciclo_completo_not_blows_up(self, capsys):
        log = ConsoleLogger()
        log.on_chain_start({"name": "cadena"}, {"input": "hola"}, run_id="r1")
        log.on_chain_end({"output": "adiós"}, run_id="r1")
        assert capsys.readouterr().out

    def test_a_error_records_without_propagarse(self):
        log = ConsoleLogger()
        log.on_chain_start({"name": "cadena"}, {}, run_id="r1")
        log.on_chain_error(RuntimeError("algo falló"), run_id="r1")

    def test_a_run_id_desconocido_at_cerrar_not_blows_up(self):
        """Callbacks arrive out of order; closing something never opened happens."""
        ConsoleLogger().on_chain_end({"output": "x"}, run_id="jamas-visto")

    def test_entradas_not_serializables_not_revientan(self):
        class Raro:
            def __repr__(self): raise ValueError("ni repr")
        log = ConsoleLogger()
        try:
            log.on_chain_start({"name": "c"}, {"raro": Raro()}, run_id="r1")
        except ValueError:
            pytest.fail("el logger propagó un fallo de formateo")


# ══════════════════════════════════════════════════════════════════════════
# Screens and chat sessions
# ══════════════════════════════════════════════════════════════════════════

class TestScreens:

    def test_the_home_exige_sesion(self, client):
        assert client.get("/", follow_redirects=False).status_code in (302, 401)

    def test_health_is_publico(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    def test_check_status_responds(self, auth_client):
        r = auth_client.get("/check_status")
        assert r.status_code == 200
        assert r.get_json()["status"] == "READY"

    def test_crear_a_chat_it_asocia_at_user(self, auth_client, db, test_user):
        from app.models import ChatSession
        before = ChatSession.query.filter_by(user_id=test_user.id).count()
        auth_client.post("/create_chat", follow_redirects=False)
        assert ChatSession.query.filter_by(user_id=test_user.id).count() == before + 1

    def test_not_can_borrar_the_chat_of_other(self, auth_client, db, test_chat_session):
        """
        The session is ALWAYS looked up filtering by user_id. Without that,
        knowing a
        id ajeno bastaría para borrar la conversación de otro.
        """
        from app.models import ChatSession, User
        otro = User(username="ajeno", role="user")
        otro.set_password("x")
        db.session.add(otro)
        db.session.commit()
        suya = ChatSession(name="privada", user_id=otro.id)
        db.session.add(suya)
        db.session.commit()

        r = auth_client.post(f"/delete_chat/{suya.id}", follow_redirects=False)
        assert r.status_code == 404
        assert ChatSession.query.get(suya.id) is not None
