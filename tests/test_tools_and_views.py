"""
test_tools_and_views.py — lo que quedaba sin cubrir: el guardarraíl de
departamento del análisis de Excel, el agente SQL, el logger de consola, la
búsqueda web, el resumidor y las pantallas.

El bloque que más importa es el primero. `ExcelAnalysisTool._find_excel_files`
es un control de ACCESO, no una comodidad: sin él, la herramienta podía leer
cualquier .xlsx del corpus saltándose el filtro por departamento que sí aplica
el retrieval documental. Un guardarraíl que solo existe en una de las dos vías
no es un guardarraíl.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from app.rag_logic.agent_intermedios import (
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
    """knowledge_base con un .xlsx en tres departamentos distintos."""
    for dep in ("compensation_benefits", "recruitment_talent", "it_workplace_policies"):
        d = tmp_path / dep
        d.mkdir()
        (d / f"{dep}_datos.xlsx").write_bytes(b"fake xlsx")
    (tmp_path / "raiz.xlsx").write_bytes(b"fake xlsx")
    return tmp_path


# ══════════════════════════════════════════════════════════════════════════
# Guardarraíl de acceso del análisis de Excel
# ══════════════════════════════════════════════════════════════════════════

class TestAccesoAFicherosExcel:

    def _tool(self, corpus, permitidos):
        return ExcelAnalysisTool(
            doc_path=str(corpus), model_name="gpt-4o-mini", allowed_departments=permitidos,
        )

    def test_none_ve_todo_solo_para_admin(self, corpus):
        # None lo produce User.get_allowed_departments() únicamente si role=admin.
        ficheros = self._tool(corpus, None)._find_excel_files()
        assert len(ficheros) == 4

    def test_una_lista_restringe_a_esos_departamentos(self, corpus):
        ficheros = self._tool(corpus, ["compensation_benefits"])._find_excel_files()
        assert len(ficheros) == 1
        assert "compensation_benefits" in ficheros[0]

    def test_no_se_cuela_el_excel_de_otro_departamento(self, corpus):
        """La propiedad que importa: lo prohibido no aparece."""
        ficheros = self._tool(corpus, ["recruitment_talent"])._find_excel_files()
        assert not any("compensation_benefits" in f for f in ficheros)

    def test_lista_vacia_no_ve_ningun_fichero(self, corpus):
        """
        Fail-closed, igual que el RBAC documental: sin departamentos asignados
        no se ve nada, en vez de verse todo.
        """
        assert self._tool(corpus, [])._find_excel_files() == []

    def test_varios_departamentos_permitidos(self, corpus):
        ficheros = self._tool(corpus, ["compensation_benefits", "recruitment_talent"])._find_excel_files()
        assert len(ficheros) == 2

    def test_se_ignoran_los_temporales_de_excel(self, corpus):
        (corpus / "compensation_benefits" / "~$borrador.xlsx").write_bytes(b"tmp")
        ficheros = self._tool(corpus, ["compensation_benefits"])._find_excel_files()
        assert not any("~$" in f for f in ficheros)

    def test_una_carpeta_inexistente_no_revienta(self, tmp_path):
        tool = ExcelAnalysisTool(doc_path=str(tmp_path / "no_existe"),
                                 model_name="gpt-4o-mini", allowed_departments=None)
        assert tool._find_excel_files() == []

    def test_sin_ficheros_lo_dice_en_vez_de_fallar(self, tmp_path):
        tool = ExcelAnalysisTool(doc_path=str(tmp_path), model_name="gpt-4o-mini",
                                 allowed_departments=None)
        assert "No se encontraron archivos Excel" in tool._run("cualquier pregunta")


class TestLimpiezaDeDataFrame:

    def test_quita_columnas_y_filas_vacias(self):
        import pandas as pd
        tool = ExcelAnalysisTool(doc_path="/x", model_name="m")
        df = pd.DataFrame({"a": [1, None], "vacia": [None, None]})
        limpio = tool._preprocess_dataframe(df)
        assert "vacia" not in limpio.columns

    def test_las_columnas_unnamed_se_vacian(self):
        import pandas as pd
        tool = ExcelAnalysisTool(doc_path="/x", model_name="m")
        df = pd.DataFrame({"Unnamed: 0": [1, 2], "real": [3, 4]})
        assert "Unnamed: 0" not in tool._preprocess_dataframe(df).columns


# ══════════════════════════════════════════════════════════════════════════
# Agente SQL (ya sin llamada al LLM)
# ══════════════════════════════════════════════════════════════════════════

class TestAgenteSQL:

    def test_no_construye_cliente_llm(self):
        """
        Su pasada de reformulación se eliminó tras comprobar que la salida la
        descartaba el agente de formato. Construir el cliente igualmente sería
        pagar un objeto por petición para nada.
        """
        tool = MagicMock()
        tool.name = "query_hr_database"
        with patch("app.rag_logic.agent_intermedios.get_llm") as get_llm:
            agente = SQLAgent(tool, model_name="gpt-4o")
            get_llm.assert_not_called()
        assert not hasattr(agente, "llm")

    def test_conserva_la_salida_bruta(self):
        tool = MagicMock()
        tool.name = "query_hr_database"
        tool.run.return_value = {"answer": "TABLA + interpretación", "source_documents": []}
        r = SQLAgent(tool, model_name="gpt-4o").run("q")
        assert r["sql_raw_output"] == "TABLA + interpretación"
        assert r["answer"] == r["sql_raw_output"]

    def test_normaliza_una_respuesta_en_texto(self):
        assert _normalize_result("solo texto") == {"answer": "solo texto", "source_documents": []}

    def test_completa_las_claves_que_falten(self):
        r = _normalize_result({"answer": "x"})
        assert r["source_documents"] == []

    def test_el_contexto_incluye_la_fuente(self):
        docs = [Document(page_content="contenido", metadata={"source": "politica.pdf"})]
        assert "politica.pdf" in _build_context_from_docs(docs)

    def test_el_contexto_admite_dicts_serializados(self):
        docs = [{"page_content": "contenido", "metadata": {"source": "a.pdf"}}]
        assert "contenido" in _build_context_from_docs(docs)

    def test_sin_documentos_devuelve_cadena_vacia(self):
        assert _build_context_from_docs([]) == ""


# ══════════════════════════════════════════════════════════════════════════
# Resumidor
# ══════════════════════════════════════════════════════════════════════════

class TestResumidor:

    def test_el_muestreo_conserva_los_extremos(self):
        """
        Al recortar para no desbordar el contexto, quedarse con los primeros N
        perdería el final del documento. El muestreo reparte.
        """
        docs = [Document(page_content=f"d{i}") for i in range(100)]
        m = _downsample_evenly(docs, 10)
        assert len(m) == 10
        assert m[0].page_content == "d0"

    def test_si_caben_todos_no_se_toca_nada(self):
        docs = [Document(page_content=f"d{i}") for i in range(5)]
        assert _downsample_evenly(docs, 10) == docs

    def test_decora_con_la_procedencia(self):
        docs = [Document(page_content="texto", metadata={"filename": "a.pdf", "page_number": 2})]
        assert "a.pdf" in _decorate_docs_with_source(docs)[0].page_content

    def test_metadata_vacia_no_revienta(self):
        assert isinstance(_meta_text({}), str)


# ══════════════════════════════════════════════════════════════════════════
# Búsqueda web
# ══════════════════════════════════════════════════════════════════════════

class TestBusquedaWeb:

    def test_devuelve_el_formato_comun(self):
        with patch("app.rag_logic.web_search.DuckDuckGoSearchRun") as ddg:
            ddg.return_value.invoke.return_value = "resultado de la búsqueda"
            r = WebSearchTool()._run("capital de Francia")
        assert r == {"answer": "resultado de la búsqueda", "source_documents": []}

    def test_un_fallo_de_red_no_tumba_la_consulta(self):
        """
        Es la única herramienta que depende de un tercero fuera de nuestro
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

class TestLoggerDeConsola:
    """
    Es un callback: se pasa a TODAS las llamadas del sistema. Un fallo suyo
    tumbaría la petición entera, así que lo que hay que probar es que no puede.
    """

    def test_un_ciclo_completo_no_revienta(self, capsys):
        log = ConsoleLogger()
        log.on_chain_start({"name": "cadena"}, {"input": "hola"}, run_id="r1")
        log.on_chain_end({"output": "adiós"}, run_id="r1")
        assert capsys.readouterr().out

    def test_un_error_se_registra_sin_propagarse(self):
        log = ConsoleLogger()
        log.on_chain_start({"name": "cadena"}, {}, run_id="r1")
        log.on_chain_error(RuntimeError("algo falló"), run_id="r1")

    def test_un_run_id_desconocido_al_cerrar_no_revienta(self):
        """Los callbacks llegan desordenados; cerrar algo que no se abrió pasa."""
        ConsoleLogger().on_chain_end({"output": "x"}, run_id="jamas-visto")

    def test_entradas_no_serializables_no_revientan(self):
        class Raro:
            def __repr__(self): raise ValueError("ni repr")
        log = ConsoleLogger()
        try:
            log.on_chain_start({"name": "c"}, {"raro": Raro()}, run_id="r1")
        except ValueError:
            pytest.fail("el logger propagó un fallo de formateo")


# ══════════════════════════════════════════════════════════════════════════
# Pantallas y sesiones de chat
# ══════════════════════════════════════════════════════════════════════════

class TestPantallas:

    def test_la_home_exige_sesion(self, client):
        assert client.get("/", follow_redirects=False).status_code in (302, 401)

    def test_health_es_publico(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    def test_check_status_responde(self, auth_client):
        r = auth_client.get("/check_status")
        assert r.status_code == 200
        assert r.get_json()["status"] == "READY"

    def test_crear_un_chat_lo_asocia_al_usuario(self, auth_client, db, test_user):
        from app.models import ChatSession
        antes = ChatSession.query.filter_by(user_id=test_user.id).count()
        auth_client.post("/create_chat", follow_redirects=False)
        assert ChatSession.query.filter_by(user_id=test_user.id).count() == antes + 1

    def test_no_se_puede_borrar_el_chat_de_otro(self, auth_client, db, test_chat_session):
        """
        La sesión se busca SIEMPRE filtrando por user_id. Sin eso, conocer un
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
