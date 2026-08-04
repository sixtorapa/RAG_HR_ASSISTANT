"""
test_pipeline.py — lo que queda entre el endpoint y las herramientas:
los guardarraíles a nivel de petición, el despacho de tools, el agente de
formato final y el índice BM25.

De los guardarraíles ya se prueba la lógica en test_guardrails.py. Aquí se
prueba lo otro, que es lo que de verdad protege: que CORTEN, y que corten
ANTES de llamar al modelo y antes de escribir nada en la base de datos.
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
# Guardarraíles: que corten, y que corten a tiempo
# ══════════════════════════════════════════════════════════════════════════

class TestCuotaDiaria:

    def test_sin_variable_no_hay_cuota(self, app, monkeypatch):
        """Railway no la necesita: OpenAI es prepago y se frena solo."""
        monkeypatch.delenv("DAILY_QUESTION_LIMIT", raising=False)
        with app.test_request_context():
            assert _quota_block() is None

    def test_cero_desactiva_la_cuota(self, app, monkeypatch):
        monkeypatch.setenv("DAILY_QUESTION_LIMIT", "0")
        with app.test_request_context():
            assert _quota_block() is None

    def test_por_debajo_del_tope_deja_pasar(self, app, db, test_user, test_chat_session, monkeypatch):
        monkeypatch.setenv("DAILY_QUESTION_LIMIT", "5")
        db.session.add(Message(session_id=test_chat_session.id, user_id=test_user.id,
                               sender="user", content="una"))
        db.session.commit()
        with app.test_request_context():
            with patch("app.main.guards.current_user", test_user):
                assert _quota_block() is None

    def test_al_alcanzar_el_tope_devuelve_429(self, app, db, test_user, test_chat_session, monkeypatch):
        monkeypatch.setenv("DAILY_QUESTION_LIMIT", "2")
        for i in range(2):
            db.session.add(Message(session_id=test_chat_session.id, user_id=test_user.id,
                                   sender="user", content=f"p{i}"))
        db.session.commit()
        with app.test_request_context():
            with patch("app.main.guards.current_user", test_user):
                blocked = _quota_block()
        assert blocked is not None
        cuerpo, codigo = blocked
        assert codigo == 429
        assert cuerpo.get_json()["quota_limit"] == 2

    def test_solo_cuentan_los_mensajes_del_usuario(self, app, db, test_user, test_chat_session, monkeypatch):
        """Las respuestas del bot no consumen cuota."""
        monkeypatch.setenv("DAILY_QUESTION_LIMIT", "2")
        for s in ("user", "bot", "bot"):
            db.session.add(Message(session_id=test_chat_session.id, user_id=test_user.id,
                                   sender=s, content="x"))
        db.session.commit()
        with app.test_request_context():
            with patch("app.main.guards.current_user", test_user):
                assert _quota_block() is None


class TestDLPEnElEndpoint:

    def test_texto_limpio_pasa(self, app):
        with app.test_request_context():
            assert _dlp_block("¿cuántos días de vacaciones tengo?", "ctx") is None

    def test_un_iban_devuelve_400(self, app):
        with app.test_request_context():
            blocked = _dlp_block("mi cuenta es ES9121000418450200051332", "ctx")
        assert blocked is not None
        cuerpo, codigo = blocked
        assert codigo == 400

    def test_el_mensaje_de_error_no_repite_el_dato(self, app):
        """Devolver el value detectado en la respuesta sería filtrarlo igual."""
        iban = "ES9121000418450200051332"
        with app.test_request_context():
            cuerpo, _ = _dlp_block(f"cuenta {iban}", "ctx")
        assert iban not in cuerpo.get_json()["error"]


class TestOrdenDeLosGuardarrailes:

    def test_el_endpoint_corta_antes_de_construir_nada(self, auth_client, test_chat_session, monkeypatch):
        """
        La propiedad que importa: con PII en la question, ni se construyen las
        herramientas ni se llama al router. Si el dato llega al modelo o a la
        base de datos, ya ha salido del perímetro.
        """
        monkeypatch.delenv("DAILY_QUESTION_LIMIT", raising=False)
        with patch("app.main.pipeline._ToolBox") as caja, \
             patch("app.main.pipeline.AgentRouter") as router:
            r = auth_client.post(
                f"/ask/{test_chat_session.id}",
                json={"question": "mi IBAN es ES9121000418450200051332"},
            )
        assert r.status_code == 400
        caja.assert_not_called()
        router.assert_not_called()

    def test_una_pregunta_bloqueada_no_se_persiste(self, auth_client, db, test_chat_session, monkeypatch):
        monkeypatch.delenv("DAILY_QUESTION_LIMIT", raising=False)
        antes = Message.query.filter_by(session_id=test_chat_session.id).count()
        auth_client.post(
            f"/ask/{test_chat_session.id}",
            json={"question": "mi IBAN es ES9121000418450200051332"},
        )
        assert Message.query.filter_by(session_id=test_chat_session.id).count() == antes


# ══════════════════════════════════════════════════════════════════════════
# Overrides y despacho de herramientas
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def caja():
    box = MagicMock()
    box.docs.name = "chat_with_documents"
    box.summary.name = "summarise_document"
    box.sql.name = "query_hr_database"
    box.excel.name = "analista_de_excel"
    box.web = None
    return box


class TestOverrides:

    def test_sin_prefijo_decide_el_router(self, caja):
        calls, _ = _calls_from_override("cuál es la política de teletrabajo", caja)
        assert calls is None

    def test_el_prefijo_sql_produce_la_misma_forma_que_el_router(self, caja):
        """
        Un override no es un camino distinto: es saltarse la decisión. Produce
        la misma lista de tool_calls, así que después se ejecuta lo mismo.
        """
        calls, limpia = _calls_from_override("SQL: cuántos empleados hay", caja)
        assert calls == [{"name": "query_hr_database", "args": {"query": "cuántos empleados hay"}}]
        assert limpia == "cuántos empleados hay"

    def test_ambas_encadena_sql_y_documentos(self, caja):
        calls, _ = _calls_from_override("AMBAS - salarios y política", caja)
        assert [c["name"] for c in calls] == ["query_hr_database", "chat_with_documents"]

    def test_pedir_un_resumen_va_a_la_herramienta_de_resumen(self, caja):
        calls, _ = _calls_from_override("dame un resumen del handbook", caja)
        assert calls[0]["name"] == "summarise_document"


class TestDespachoDeHerramientas:

    def test_una_herramienta_desconocida_no_revienta(self, caja):
        r = _run_tools([{"name": "no_existe", "args": {}}], caja, "q", [])
        assert "No sé qué herramienta usar" in r[0]["answer"]
        assert r[0]["source_documents"] == []

    def test_el_resultado_siempre_es_un_dict(self, caja):
        """Las tools pueden devolver str; aguas abajo se asume dict."""
        caja.summary.run.return_value = "una cadena suelta"
        r = _run_tools([{"name": "summarise_document", "args": {}}], caja, "q", [])
        assert isinstance(r[0], dict)
        assert r[0]["answer"] == "una cadena suelta"

    def test_se_etiqueta_el_origen_de_cada_paso(self, caja):
        caja.summary.run.return_value = {"answer": "x", "source_documents": []}
        r = _run_tools([{"name": "summarise_document", "args": {}}], caja, "q", [])
        assert r[0]["origin"] == "summarise_document"

    def test_el_resultado_sql_se_encadena_a_la_consulta_documental(self, caja):
        """El modo híbrido: lo que devuelve SQL entra como context en DOCS."""
        caja.sql_agent.run.return_value = {"answer": "t", "sql_raw_output": "dept | n\nEng | 12",
                                           "source_documents": []}
        caja.docs.run.return_value = {"answer": "según la política...", "source_documents": []}
        _run_tools([{"name": "query_hr_database", "args": {"query": "q"}},
                    {"name": "chat_with_documents", "args": {"question": "q"}}], caja, "q", [])
        question = caja.docs.run.call_args[0][0]["question"]
        assert "SALIDA SQL" in question

    def test_sin_llamadas_devuelve_lista_vacia(self, caja):
        assert _run_tools([], caja, "q", []) == []
        assert _run_tools(None, caja, "q", []) == []


class TestContextoSQL:

    def test_prefiere_la_salida_bruta(self):
        doc = _sql_context_document({"sql_raw_output": "TABLA", "answer": "prosa"})
        assert "TABLA" in doc.page_content

    def test_recorta_las_salidas_largas(self):
        doc = _sql_context_document({"sql_raw_output": "\n".join(f"row {i}" for i in range(200))})
        assert len(doc.page_content.splitlines()) < 40

    def test_sin_contenido_no_produce_documento(self):
        assert _sql_context_document({"sql_raw_output": "", "answer": ""}) is None


# ══════════════════════════════════════════════════════════════════════════
# Agente de formato final
# ══════════════════════════════════════════════════════════════════════════

class TestAgenteDeFormato:

    def test_prefiere_la_salida_bruta_de_sql(self):
        """
        Este detalle es el que hizo que la reformulación de SQLAgent fuese
        código muerto: su `answer` no se leía nunca.
        """
        text = _build_contributions_summary([
            {"origin": "query_hr_database", "sql_raw_output": "TABLA CRUDA", "answer": "REFORMULADO"},
        ])
        assert "TABLA CRUDA" in text
        assert "REFORMULADO" not in text

    def test_usa_answer_cuando_no_hay_salida_sql(self):
        text = _build_contributions_summary([{"origin": "chat_with_documents", "answer": "la respuesta"}])
        assert "la respuesta" in text

    def test_trunca_los_bloques_enormes(self):
        text = _build_contributions_summary([{"origin": "x", "answer": "y" * 20000}])
        assert "TRUNCATED" in text

    def test_sin_contribuciones_lo_dice(self):
        assert "No useful response" in _build_contributions_summary([])

    def test_fusiona_las_fuentes_de_todos_los_pasos(self):
        docs = _merge_source_docs([
            {"source_documents": [Document(page_content="a")]},
            {"source_documents": [Document(page_content="b")]},
            {"source_documents": []},
        ])
        assert [d.page_content for d in docs] == ["a", "b"]

    def test_devuelve_respuesta_y_fuentes(self):
        with patch("app.rag_logic.agent_reasoning.get_llm") as get_llm:
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="  respuesta final  ")
            get_llm.return_value = llm
            r = ReasoningAgent(model_name="gpt-4o-mini").run(
                "question", [{"origin": "x", "answer": "dato", "source_documents": [Document(page_content="f")]}],
            )
        assert r["answer"] == "respuesta final"
        assert len(r["source_documents"]) == 1


# ══════════════════════════════════════════════════════════════════════════
# Índice BM25 persistido
# ══════════════════════════════════════════════════════════════════════════

class TestBM25:

    def test_construye_desde_textos_y_metadata(self):
        r = build_bm25_retriever(["text uno", "text dos"], [{"a": 1}, {"a": 2}])
        assert r is not None

    def test_sin_documentos_devuelve_none(self):
        assert build_bm25_retriever([], []) is None

    def test_longitudes_desiguales_devuelven_none(self):
        """Un zip() silencioso perdería documentos sin avisar."""
        assert build_bm25_retriever(["a", "b"], [{"x": 1}]) is None

    def test_ida_y_vuelta_por_disco(self, tmp_path):
        vs = MagicMock()
        vs.get.return_value = {"documents": ["política de vacaciones", "guía de onboarding"],
                               "metadatas": [{"f": "a.pdf"}, {"f": "b.pdf"}]}
        assert persist_bm25_index(vs, str(tmp_path)) is True
        assert load_bm25_index(str(tmp_path)) is not None

    def test_sin_fichero_devuelve_none(self, tmp_path):
        assert load_bm25_index(str(tmp_path)) is None

    def test_un_pickle_corrupto_no_tumba_la_consulta(self, tmp_path):
        """
        Un pickle antiguo puede no abrirse tras actualizar una librería. La
        consulta debe seguir funcionando sin la pata léxica, no reventar.
        """
        (tmp_path / "_bm25_index.pkl").write_bytes(b"esto no es un pickle")
        assert load_bm25_index(str(tmp_path)) is None

    def test_un_vector_store_vacio_no_persiste_nada(self, tmp_path):
        vs = MagicMock()
        vs.get.return_value = {"documents": [], "metadatas": []}
        assert persist_bm25_index(vs, str(tmp_path)) is False
