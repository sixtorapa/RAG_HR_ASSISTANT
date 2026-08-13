"""
test_retrieval.py — the provider factory and the two retrieval pieces that
carry the most decisions: the granularity filter and child -> parent expansion.

None of this calls a model or opens an index: what is checked are the
decisions, which is where the failures that matter live.
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
# Provider factory
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def no_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    yield


class TestProviderFactory:

    def test_the_default_is_openai(self, no_provider):
        """
        The default protects the existing deployment: deploying without setting
        the variable behaves exactly as it did before.
        """
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
            openai.assert_called_once()

    @pytest.mark.parametrize("value", ["bedrock", "BEDROCK", "  Bedrock  ", "'bedrock'", '"bedrock"'])
    def test_tolera_as_llegue_the_variable(self, monkeypatch, value):
        """
        An environment variable arrives from .env, from the Railway panel, from
        consola de Lambda o de un export: cada uno la trata distinto.
        """
        monkeypatch.setenv("LLM_PROVIDER", value)
        with patch.object(llm_factory, "ChatBedrockConverse") as bedrock:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
            bedrock.assert_called_once()

    def test_only_spaces_cae_at_default(self, monkeypatch):
        """Normalise and THEN apply the default: '   ' is truthy."""
        monkeypatch.setenv("LLM_PROVIDER", "   ")
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
            openai.assert_called_once()

    def test_a_provider_desconocido_avisa_and_usa_openai(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0)
            openai.assert_called_once()
        assert "not recognised" in caplog.text

    def test_traduce_the_model_a_a_perfil_of_inferencia_europeo(self, monkeypatch):
        """
        Bedrock only offers these models as INFERENCE_PROFILE. With a bare
        secas responde ValidationException, y el prefijo `eu.` mantiene la
        inferencia dentro de la UE — que en un sistema con datos de personal
        es el mismo argumento que el RBAC.
        """
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with patch.object(llm_factory, "ChatBedrockConverse") as bedrock:
            llm_factory.get_llm("gpt-4o", 0.0)
            assert bedrock.call_args.kwargs["model_id"].startswith("eu.")

    def test_an_unmapped_model_raises_with_a_named_error(self, monkeypatch):
        """
        Falling back to a default model would make "I asked for this model" and
        "me dieron otro" — el mismo fallo silencioso que se corrigió en
        cost_calculator.
        """
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with pytest.raises(ValueError, match="gpt-5-turbo"):
            llm_factory.get_llm("gpt-5-turbo", 0.0)

    def test_a_bedrock_id_passes_through_unchanged(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with patch.object(llm_factory, "ChatBedrockConverse") as bedrock:
            llm_factory.get_llm("eu.anthropic.claude-sonnet-4-6", 0.0)
            assert bedrock.call_args.kwargs["model_id"] == "eu.anthropic.claude-sonnet-4-6"

    def test_the_embeddings_ignoran_the_provider(self, monkeypatch):
        """
        A deliberate decision: changing the embedding model would invalidate the
        índice entero y las métricas ya medidas. La generación se migra; el
        espacio vectorial no se toca.
        """
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        with patch.object(llm_factory, "OpenAIEmbeddings") as emb:
            llm_factory.get_embeddings()
            emb.assert_called_once()

    def test_forwards_kwargs(self, no_provider):
        """Without this, `callbacks` would be dropped silently and logging would stop."""
        cb = [MagicMock()]
        with patch.object(llm_factory, "ChatOpenAI") as openai:
            llm_factory.get_llm("gpt-4o-mini", 0.0, callbacks=cb)
            assert openai.call_args.kwargs["callbacks"] is cb

    def test_the_provider_is_read_on_every_call(self, monkeypatch):
        """Not frozen at import: changing it does not require a restart."""
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

class TestGranularityFilter:

    def test_by_default_searches_by_the_children(self, monkeypatch):
        monkeypatch.delenv("RETRIEVAL_CHUNK_TYPE", raising=False)
        assert _chunk_type_filter() == {"chunk_type": "micro"}

    @pytest.mark.parametrize("value", ["all", "any", "  ALL  "])
    def test_all_desactiva_the_filter(self, monkeypatch, value):
        monkeypatch.setenv("RETRIEVAL_CHUNK_TYPE", value)
        assert _chunk_type_filter() is None

    def test_macro_searches_by_the_parents(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CHUNK_TYPE", "macro")
        assert _chunk_type_filter() == {"chunk_type": "macro"}

    def test_a_value_desconocido_cae_a_micro(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_CHUNK_TYPE", "mediano")
        assert _chunk_type_filter() == {"chunk_type": "micro"}

    def test_the_expansion_can_apagar(self, monkeypatch):
        monkeypatch.setenv("PARENT_EXPANSION", "0")
        assert _parent_expansion_enabled() is False
        monkeypatch.setenv("PARENT_EXPANSION", "1")
        assert _parent_expansion_enabled() is True


# ══════════════════════════════════════════════════════════════════════════
# Child -> parent expansion
# ══════════════════════════════════════════════════════════════════════════

def _hijo(chunk_id, parent_id):
    return Document(
        page_content=f"text del hijo {chunk_id}",
        metadata={"chunk_id": chunk_id, "parent_chunk_id": parent_id, "chunk_type": "micro"},
    )


def _construir(base, vs, max_docs=12):
    """
    `ParentExpansionRetriever` is a Pydantic model and its fields are typed
    (`BaseRetriever`, `Chroma`), así que la validación rechaza un mock.
    `model_construct` la salta en el test y deja el contrato del código
    strictly: loosening the annotation to `Any` just to make a test pass would
    be making the code worse in order to test it.
    """
    return ParentExpansionRetriever.model_construct(
        base_retriever=base, vector_store=vs, max_docs=max_docs,
    )


def _almacen(padres):
    """Fake Chroma: returns the parent whose chunk_id is requested."""
    vs = MagicMock()

    def get(where=None, include=None, limit=None):
        pid = (where or {}).get("chunk_id")
        if pid in padres:
            return {"documents": [padres[pid]], "metadatas": [{"chunk_id": pid, "chunk_type": "macro"}]}
        return {"documents": [], "metadatas": []}

    vs.get.side_effect = get
    return vs


class TestParentExpansion:

    def test_several_children_of_the_same_parent_collapse_in_one(self):
        """
        The property that removes the redundancy: three overlapping fragments of
        the same page used to reach the prompt as three entries.
        """
        base = MagicMock()
        base.get_relevant_documents.return_value = [
            _hijo("h1", "P1"), _hijo("h2", "P1"), _hijo("h3", "P1"),
        ]
        r = _construir(base, _almacen({"P1": "página 1 entera"}))
        out = r.get_relevant_documents("cualquier cosa")
        assert len(out) == 1
        assert out[0].page_content == "página 1 entera"

    def test_parents_distintos_conservan_all(self):
        base = MagicMock()
        base.get_relevant_documents.return_value = [_hijo("h1", "P1"), _hijo("h2", "P2")]
        r = _construir(base, _almacen({"P1": "página 1", "P2": "página 2"}))
        assert [d.page_content for d in r.get_relevant_documents("q")] == ["página 1", "página 2"]

    def test_keeps_the_orden_of_the_retriever_of_abajo(self):
        """
        The parent takes the position of its FIRST child, so the ranking of the
        híbrido no se pierde por el camino.
        """
        base = MagicMock()
        base.get_relevant_documents.return_value = [
            _hijo("h1", "P2"), _hijo("h2", "P1"), _hijo("h3", "P2"),
        ]
        r = _construir(base, _almacen({"P1": "página 1", "P2": "página 2"}))
        assert [d.page_content for d in r.get_relevant_documents("q")] == ["página 2", "página 1"]

    def test_when_the_parent_is_missing_the_child_is_returned(self):
        """Degradation: with no parent in the index, the child beats nothing."""
        base = MagicMock()
        base.get_relevant_documents.return_value = [_hijo("h1", "DESAPARECIDO")]
        r = _construir(base, _almacen({}))
        out = r.get_relevant_documents("q")
        assert len(out) == 1
        assert out[0].page_content == "text del hijo h1"

    def test_a_document_without_a_parent_passes_through_unchanged(self):
        suelto = Document(page_content="chunk de un .md", metadata={"chunk_id": "s1"})
        base = MagicMock()
        base.get_relevant_documents.return_value = [suelto]
        r = _construir(base, _almacen({}))
        assert r.get_relevant_documents("q")[0].page_content == "chunk de un .md"

    def test_respeta_the_cap_of_documents(self):
        base = MagicMock()
        base.get_relevant_documents.return_value = [_hijo(f"h{i}", f"P{i}") for i in range(20)]
        padres = {f"P{i}": f"página {i}" for i in range(20)}
        r = _construir(base, _almacen(padres), max_docs=5)
        assert len(r.get_relevant_documents("q")) == 5

    def test_without_results_not_blows_up(self):
        base = MagicMock()
        base.get_relevant_documents.return_value = []
        r = _construir(base, _almacen({}))
        assert r.get_relevant_documents("q") == []

    def test_a_fallo_of_the_store_not_brings_down_the_query(self):
        base = MagicMock()
        base.get_relevant_documents.return_value = [_hijo("h1", "P1")]
        vs = MagicMock()
        vs.get.side_effect = RuntimeError("chroma caído")
        r = _construir(base, vs)
        out = r.get_relevant_documents("q")
        assert out[0].page_content == "text del hijo h1"
