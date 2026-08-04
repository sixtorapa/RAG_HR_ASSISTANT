"""
test_retrieval.py — la factoría de provider y las dos piezas nuevas del
retrieval: el filtro de granularidad y la expansión hijo -> padre.

Nada de esto llama a un modelo ni abre un índice: se comprueban las decisiones,
que es donde están los fallos que importan.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from app.rag_logic import llm_factory
from app.rag_logic.qa_chain import (
    ParentExpansionRetriever,
    _chunk_type_filter,
    _parent_expansion_enabled,
)


# ══════════════════════════════════════════════════════════════════════════
# Factoría de provider
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sin_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    yield


class TestFactoriaDeProveedor:

    def test_el_defecto_es_openai(self, sin_provider):
        """
        El defecto protege el despliegue existente: desplegar sin definir la
        variable se comporta igual que antes de que la factoría existiera.
        """
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
            openai.assert_called_once()

    @pytest.mark.parametrize("value", ["bedrock", "BEDROCK", "  Bedrock  ", "'bedrock'", '"bedrock"'])
    def test_tolera_como_llegue_la_variable(self, monkeypatch, value):
        """
        Una variable de entorno llega del .env, del panel de Railway, de la
        consola de Lambda o de un export: cada uno la trata distinto.
        """
        monkeypatch.setenv("LLM_PROVIDER", value)
        with patch.object(llm_factory, "ChatBedrockConverse") as bedrock:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
            bedrock.assert_called_once()

    def test_solo_espacios_cae_al_defecto(self, monkeypatch):
        """Normalizar y DESPUÉS aplicar el defecto: '   ' es truthy."""
        monkeypatch.setenv("LLM_PROVIDER", "   ")
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
            openai.assert_called_once()

    def test_un_proveedor_desconocido_avisa_y_usa_openai(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
            openai.assert_called_once()
        assert "no reconocido" in caplog.text

    def test_traduce_el_modelo_a_un_perfil_de_inferencia_europeo(self, monkeypatch):
        """
        Bedrock solo ofrece estos modelos como INFERENCE_PROFILE. Con el ID a
        secas responde ValidationException, y el prefijo `eu.` mantiene la
        inferencia dentro de la UE — que en un sistema con datos de personal
        es el mismo argumento que el RBAC.
        """
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with patch.object(llm_factory, "ChatBedrockConverse") as bedrock:
            llm_factory.get_llm("gpt-4o", 0.0)
            assert bedrock.call_args.kwargs["model_id"].startswith("eu.")

    def test_un_modelo_sin_equivalente_revienta_con_nombre_y_apellido(self, monkeypatch):
        """
        Caer a un modelo por defecto haría indistinguible "pedí este modelo" de
        "me dieron otro" — el mismo fallo silencioso que se corrigió en
        cost_calculator.
        """
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with pytest.raises(ValueError, match="gpt-5-turbo"):
            llm_factory.get_llm("gpt-5-turbo", 0.0)

    def test_un_id_de_bedrock_pasa_tal_cual(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with patch.object(llm_factory, "ChatBedrockConverse") as bedrock:
            llm_factory.get_llm("eu.anthropic.claude-sonnet-4-6", 0.0)
            assert bedrock.call_args.kwargs["model_id"] == "eu.anthropic.claude-sonnet-4-6"

    def test_los_embeddings_ignoran_el_proveedor(self, monkeypatch):
        """
        Decisión consciente: cambiar el modelo de embeddings invalidaría el
        índice entero y las métricas ya medidas. La generación se migra; el
        espacio vectorial no se toca.
        """
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with patch.object(llm_factory, "OpenAIEmbeddings") as emb:
            llm_factory.get_embeddings()
            emb.assert_called_once()

    def test_reenvia_kwargs(self, sin_provider):
        """Sin esto, `callbacks` se perdería en silencio y no habría logging."""
        cb = [MagicMock()]
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0, callbacks=cb)
            assert openai.call_args.kwargs["callbacks"] is cb

    def test_el_proveedor_se_lee_en_cada_llamada(self, monkeypatch):
        """No se congela al importar: cambiarlo no exige reiniciar el proceso."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with patch.object(llm_factory, "ChatBedrockConverse") as bedrock:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
        assert openai.called and bedrock.called


# ══════════════════════════════════════════════════════════════════════════
# Filtro de granularidad
# ══════════════════════════════════════════════════════════════════════════

class TestFiltroDeGranularidad:

    def test_por_defecto_se_busca_por_los_hijos(self, monkeypatch):
        monkeypatch.delenv("RETRIEVAL_CHUNK_TYPE", raising=False)
        assert _chunk_type_filter() == {"chunk_type": "micro"}

    @pytest.mark.parametrize("value", ["all", "any", "  ALL  "])
    def test_all_desactiva_el_filtro(self, monkeypatch, value):
        monkeypatch.setenv("RETRIEVAL_CHUNK_TYPE", value)
        assert _chunk_type_filter() is None

    def test_macro_busca_por_los_padres(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CHUNK_TYPE", "macro")
        assert _chunk_type_filter() == {"chunk_type": "macro"}

    def test_un_valor_desconocido_cae_a_micro(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CHUNK_TYPE", "mediano")
        assert _chunk_type_filter() == {"chunk_type": "micro"}

    def test_la_expansion_se_puede_apagar(self, monkeypatch):
        monkeypatch.setenv("PARENT_EXPANSION", "0")
        assert _parent_expansion_enabled() is False
        monkeypatch.setenv("PARENT_EXPANSION", "1")
        assert _parent_expansion_enabled() is True


# ══════════════════════════════════════════════════════════════════════════
# Expansión hijo -> padre
# ══════════════════════════════════════════════════════════════════════════

def _hijo(chunk_id, parent_id):
    return Document(
        page_content=f"text del hijo {chunk_id}",
        metadata={"chunk_id": chunk_id, "parent_chunk_id": parent_id, "chunk_type": "micro"},
    )


def _construir(base, vs, max_docs=12):
    """
    `ParentExpansionRetriever` es un modelo Pydantic y sus campos están tipados
    (`BaseRetriever`, `Chroma`), así que la validación rechaza un mock.
    `model_construct` la salta en el test y deja el contrato del código
    intacto: relajar la anotación a `Any` para que pase un test sería empeorar
    el código para poder probarlo.
    """
    return ParentExpansionRetriever.model_construct(
        base_retriever=base, vector_store=vs, max_docs=max_docs,
    )


def _almacen(padres):
    """Chroma simulado: devuelve el padre cuyo chunk_id se pide."""
    vs = MagicMock()

    def get(where=None, include=None, limit=None):
        pid = (where or {}).get("chunk_id")
        if pid in padres:
            return {"documents": [padres[pid]], "metadatas": [{"chunk_id": pid, "chunk_type": "macro"}]}
        return {"documents": [], "metadatas": []}

    vs.get.side_effect = get
    return vs


class TestExpansionAlPadre:

    def test_varios_hijos_del_mismo_padre_colapsan_en_uno(self):
        """
        La propiedad que arregla la redundancia: tres fragmentos solapados de
        la misma página llegaban al prompt como tres entradas.
        """
        base = MagicMock()
        base.get_relevant_documents.return_value = [
            _hijo("h1", "P1"), _hijo("h2", "P1"), _hijo("h3", "P1"),
        ]
        r = _construir(base, _almacen({"P1": "página 1 entera"}))
        out = r.get_relevant_documents("cualquier cosa")
        assert len(out) == 1
        assert out[0].page_content == "página 1 entera"

    def test_padres_distintos_se_conservan_todos(self):
        base = MagicMock()
        base.get_relevant_documents.return_value = [_hijo("h1", "P1"), _hijo("h2", "P2")]
        r = _construir(base, _almacen({"P1": "página 1", "P2": "página 2"}))
        assert [d.page_content for d in r.get_relevant_documents("q")] == ["página 1", "página 2"]

    def test_se_conserva_el_orden_del_retriever_de_abajo(self):
        """
        El padre ocupa la posición de su PRIMER hijo, así que el ranking del
        híbrido no se pierde por el camino.
        """
        base = MagicMock()
        base.get_relevant_documents.return_value = [
            _hijo("h1", "P2"), _hijo("h2", "P1"), _hijo("h3", "P2"),
        ]
        r = _construir(base, _almacen({"P1": "página 1", "P2": "página 2"}))
        assert [d.page_content for d in r.get_relevant_documents("q")] == ["página 2", "página 1"]

    def test_si_el_padre_no_esta_se_devuelve_el_hijo(self):
        """Degradación: sin padre en el índice, mejor el hijo que nada."""
        base = MagicMock()
        base.get_relevant_documents.return_value = [_hijo("h1", "DESAPARECIDO")]
        r = _construir(base, _almacen({}))
        out = r.get_relevant_documents("q")
        assert len(out) == 1
        assert out[0].page_content == "text del hijo h1"

    def test_un_documento_sin_padre_pasa_tal_cual(self):
        suelto = Document(page_content="chunk de un .md", metadata={"chunk_id": "s1"})
        base = MagicMock()
        base.get_relevant_documents.return_value = [suelto]
        r = _construir(base, _almacen({}))
        assert r.get_relevant_documents("q")[0].page_content == "chunk de un .md"

    def test_se_respeta_el_tope_de_documentos(self):
        base = MagicMock()
        base.get_relevant_documents.return_value = [_hijo(f"h{i}", f"P{i}") for i in range(20)]
        padres = {f"P{i}": f"página {i}" for i in range(20)}
        r = _construir(base, _almacen(padres), max_docs=5)
        assert len(r.get_relevant_documents("q")) == 5

    def test_sin_resultados_no_revienta(self):
        base = MagicMock()
        base.get_relevant_documents.return_value = []
        r = _construir(base, _almacen({}))
        assert r.get_relevant_documents("q") == []

    def test_un_fallo_del_almacen_no_tumba_la_consulta(self):
        base = MagicMock()
        base.get_relevant_documents.return_value = [_hijo("h1", "P1")]
        vs = MagicMock()
        vs.get.side_effect = RuntimeError("chroma caído")
        r = _construir(base, vs)
        out = r.get_relevant_documents("q")
        assert out[0].page_content == "text del hijo h1"
