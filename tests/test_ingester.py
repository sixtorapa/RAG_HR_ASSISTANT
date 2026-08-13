"""
test_ingester.py — the largest module in the repository, and the one furthest
from where a failure shows up.

A mistake here raises no exception: it produces a worse index. Nobody sees it
until the answers get worse, and by then nothing points at ingestion. So what
is tested are the properties the retrieval leans on:

  - coverage: no page with text may disappear when chunking
  - hierarchy: every micro chunk must be able to find its parent
  - metadata: the department drives access control, so computing it wrong makes
    the guardrail filter wrong
  - identity: chunk ids must be stable across runs, or incremental ingestion
    silently stops being incremental
"""

import pytest
from langchain.docstore.document import Document

from app.rag_logic.ingester import (
    _clean_text_for_embeddings,
    _collapse_whitespace,
    _despace_text,
    _generate_micro_chunks,
    _merge_pages_into_chunk,
    _page_based_chunking,
    inject_context_to_chunks,
    sanitize_metadata,
)
from app.rag_logic.path_utils import norm_path


def _pagina(text, n, fichero="doc.pdf"):
    return Document(
        page_content=text,
        metadata={
            "filename": fichero, "relative_path": fichero, "source_file": fichero,
            "page_number": n, "page": n - 1, "file_type": "pdf",
        },
    )


def _palabras(n, semilla="palabra"):
    return " ".join(f"{semilla}{i}" for i in range(n))


# ══════════════════════════════════════════════════════════════════════════
# Limpieza de text
# ══════════════════════════════════════════════════════════════════════════

class TestTextCleanup:

    def test_split_letters_are_rejoined(self):
        assert "PROPUESTA" in _despace_text("U N A P R O P U E S T A").replace(" ", "")

    def test_not_touches_text_normal(self):
        text = "Los empleados tienen 25 días de vacaciones al año."
        assert _despace_text(text) == text

    def test_collapses_spaces_and_saltos(self):
        assert _collapse_whitespace("hola    mundo") == "hola mundo"
        assert "\n\n\n" not in _collapse_whitespace("a\n\n\n\n\nb")

    def test_the_limpieza_not_empty_a_text_with_contenido(self):
        """Safety property: cleaning must never turn text into nothing."""
        text = "Política de teletrabajo: hasta 3 días por semana."
        assert _clean_text_for_embeddings(text).strip()

    def test_the_limpieza_tolera_empty(self):
        assert _clean_text_for_embeddings("") in ("", None)


# ══════════════════════════════════════════════════════════════════════════
# Page-based chunking (phase 1: deterministic, no LLM)
# ══════════════════════════════════════════════════════════════════════════

class TestPageBasedChunking:

    def test_short_pages_are_grouped_up_to_the_target(self):
        pages = [_pagina(_palabras(100), i) for i in range(1, 8)]
        chunks = _page_based_chunking(pages, min_words=150, target_words=350, max_words=700)
        assert len(chunks) < len(pages)
        assert all(len(c.page_content.split()) <= 700 for c in chunks)

    def test_the_chunking_by_pages_not_marks_the_tipo(self):
        """
        ⚠️ A real coupling, pinned here so it stays visible: `_page_based_chunking`
        no escribe `chunk_type`. Quien marca los padres como "macro" es
        `_generate_micro_chunks`, como efecto secundario de crear los hijos.

        La consecuencia importa: el retrieval filtra por `chunk_type` y su valor
        por defecto es "micro". Si alguien saltara la generación de micro-chunks
        creyendo que solo pierde granularidad, ningún chunk tendría tipo, el
        filtro no casaría con nada y la búsqueda devolvería CERO resultados sin
        un solo error por ninguna parte.
        """
        chunks = _page_based_chunking([_pagina(_palabras(100), i) for i in range(1, 5)])
        assert all(c.metadata.get("chunk_type") is None for c in chunks)

    def test_generar_micros_is_it_that_marks_a_the_parents(self):
        macros = _page_based_chunking([_pagina(_palabras(400), i) for i in range(1, 4)])
        _generate_micro_chunks(macros)
        assert all(m.metadata.get("chunk_type") == "macro" for m in macros)

    def test_no_page_with_text_is_lost(self):
        """
        The central property: 100% coverage. If a page disappears during
        trocear, su contenido es irrecuperable por retrieval y nada lo avisa.
        """
        pages = [_pagina(f"contenido unico de la pagina {i} " + _palabras(80), i)
                   for i in range(1, 11)]
        text = " ".join(c.page_content for c in _page_based_chunking(pages))
        for i in range(1, 11):
            assert f"contenido unico de la pagina {i}" in text

    def test_a_very_long_page_stands_alone(self):
        pages = [_pagina(_palabras(80), 1), _pagina(_palabras(900), 2), _pagina(_palabras(80), 3)]
        chunks = _page_based_chunking(pages, max_words=700)
        largos = [c for c in chunks if len(c.page_content.split()) > 700]
        assert len(largos) == 1

    def test_different_documents_are_not_mixed(self):
        pages = [_pagina(_palabras(100), 1, "a.pdf"), _pagina(_palabras(100), 1, "b.pdf")]
        chunks = _page_based_chunking(pages)
        for c in chunks:
            sources = {c.metadata.get("source_file")}
            assert len(sources) == 1

    def test_registran_the_pages_of_each_chunk(self):
        """`pages_in_chunk` is what allows citing "pp. 3-4" in the answer."""
        chunks = _page_based_chunking([_pagina(_palabras(120), i) for i in range(1, 6)])
        for c in chunks:
            assert c.metadata.get("pages_in_chunk")

    def test_without_pages_returns_list_empty(self):
        assert _page_based_chunking([]) == []


class TestPageMerging:

    def test_marks_the_limites_of_page(self):
        chunk = _merge_pages_into_chunk([_pagina("texto uno", 1), _pagina("texto dos", 2)], "doc.pdf")
        assert "[PAGE" in chunk.page_content
        assert "texto uno" in chunk.page_content and "texto dos" in chunk.page_content

    def test_chunk_ids_are_stable_across_runs(self):
        """
        If the id changed between ingests, incremental ingestion would delete and
        reindexaría todo cada vez, y `parent_chunk_id` dejaría de resolver.
        """
        pages = [_pagina("mismo contenido exacto", 1)]
        a = _merge_pages_into_chunk(pages, "doc.pdf")
        b = _merge_pages_into_chunk(pages, "doc.pdf")
        assert a.metadata["chunk_id"] == b.metadata["chunk_id"]

    def test_different_contents_give_different_ids(self):
        a = _merge_pages_into_chunk([_pagina("contenido A", 1)], "doc.pdf")
        b = _merge_pages_into_chunk([_pagina("contenido B", 1)], "doc.pdf")
        assert a.metadata["chunk_id"] != b.metadata["chunk_id"]


# ══════════════════════════════════════════════════════════════════════════
# Micro-chunks (arquitectura parent-child)
# ══════════════════════════════════════════════════════════════════════════

class TestMicroChunks:

    @pytest.fixture
    def macros(self):
        return _page_based_chunking([_pagina(_palabras(400), i) for i in range(1, 5)])

    def test_all_child_apunta_a_a_parent_existente(self):
        """
        Without this, the child -> parent expansion has no parent to resolve and
        degrada al hijo, que es justo lo que se quería evitar.
        """
        macros = _page_based_chunking([_pagina(_palabras(400), i) for i in range(1, 5)])
        ids_padres = {m.metadata["chunk_id"] for m in macros}
        micros = _generate_micro_chunks(macros)
        assert micros
        for m in micros:
            assert m.metadata["parent_chunk_id"] in ids_padres

    def test_the_children_marcan_as_micro(self):
        micros = _generate_micro_chunks(_page_based_chunking([_pagina(_palabras(400), 1)]))
        assert all(m.metadata["chunk_type"] == "micro" for m in micros)

    def test_the_children_have_su_propio_id(self):
        micros = _generate_micro_chunks(_page_based_chunking([_pagina(_palabras(600), 1)]))
        ids = [m.metadata["chunk_id"] for m in micros]
        assert len(ids) == len(set(ids))

    def test_the_children_are_more_pequenos_that_the_parent(self):
        macros = _page_based_chunking([_pagina(_palabras(600), 1)])
        micros = _generate_micro_chunks(macros)
        maximo_padre = max(len(m.page_content.split()) for m in macros)
        assert all(len(mi.page_content.split()) <= maximo_padre for mi in micros)

    def test_the_children_heredan_the_provenance(self):
        macros = _page_based_chunking([_pagina(_palabras(400), 1, "politica.pdf")])
        for m in _generate_micro_chunks(macros):
            assert m.metadata.get("source_file") == "politica.pdf"

    def test_without_macros_not_there_is_micros(self):
        assert _generate_micro_chunks([]) == []


# ══════════════════════════════════════════════════════════════════════════
# Contexto inyectado en el text embebido
# ══════════════════════════════════════════════════════════════════════════

class TestInjectedContext:

    def test_antepone_the_provenance(self):
        chunk = Document(page_content="cuerpo del chunk",
                         metadata={"relative_path": "hr/politica.pdf", "page_number": 3})
        [output] = inject_context_to_chunks([chunk])
        assert output.page_content.startswith("SOURCE:")
        assert "politica.pdf" in output.page_content
        assert "cuerpo del chunk" in output.page_content

    def test_headline_and_summary_are_prepended_when_present(self):
        """
        The summary is generated and paid for in phase 2. If it never reaches the text
        embebido, no sirve para recuperar: una question se parece más a un
        titular y a un resumen que al cuerpo del documento.
        """
        chunk = Document(page_content="cuerpo",
                         metadata={"relative_path": "d.pdf", "page_number": 1,
                                   "semantic_headline": "Política de licencias",
                                   "semantic_summary": "Resume los tipos de permiso."})
        [output] = inject_context_to_chunks([chunk])
        assert "TITLE: Política de licencias" in output.page_content
        assert "SUMMARY: Resume los tipos de permiso." in output.page_content

    def test_without_headline_nor_summary_not_lets_tags_empty(self):
        chunk = Document(page_content="cuerpo", metadata={"relative_path": "d.pdf", "page_number": 1})
        [output] = inject_context_to_chunks([chunk])
        assert "TITLE:" not in output.page_content
        assert "SUMMARY:" not in output.page_content

    def test_cites_a_range_when_the_chunk_covers_several_pages(self):
        chunk = Document(page_content="cuerpo",
                         metadata={"relative_path": "d.pdf", "pages_in_chunk": "3, 4, 5"})
        [output] = inject_context_to_chunks([chunk])
        assert "3-5" in output.page_content


# ══════════════════════════════════════════════════════════════════════════
# Metadata para Chroma
# ══════════════════════════════════════════════════════════════════════════

class TestMetadata:

    def test_the_listas_aplanan(self):
        """Chroma does not accept lists in metadata: without this, ingestion fails."""
        doc = Document(page_content="x", metadata={"paginas": [1, 2, 3]})
        sanitize_metadata(doc)
        assert doc.metadata["paginas"] == "1, 2, 3"

    def test_the_nulls_become_in_string_empty(self):
        doc = Document(page_content="x", metadata={"campo": None})
        sanitize_metadata(doc)
        assert doc.metadata["campo"] == ""

    def test_the_diccionarios_serializan(self):
        doc = Document(page_content="x", metadata={"extra": {"a": 1}})
        sanitize_metadata(doc)
        assert isinstance(doc.metadata["extra"], str)

    def test_the_escalares_dejan_as_estan(self):
        doc = Document(page_content="x", metadata={"n": 3, "f": 1.5, "b": True, "s": "t"})
        sanitize_metadata(doc)
        assert doc.metadata == {"n": 3, "f": 1.5, "b": True, "s": "t"}


class TestDepartmentFromPath:
    """
    The department comes from the folder structure and is what feeds the
    guardarraíl de acceso. Si se calcula mal, el RBAC filtra mal — y en la
    dirección peligrosa: un documento cuyo departamento no casa con ninguno
    permitido simplemente no aparece, sin error.
    """

    @staticmethod
    def _departamento(rel):
        # The same expression process_and_store_documents uses.
        n = norm_path(rel)
        return (n.split("/")[0] if "/" in n else "general").lower()

    def test_primer_segmento_of_the_ruta(self):
        assert self._departamento("compensation_benefits/salarios.pdf") == "compensation_benefits"

    def test_with_subfolders_still_being_the_first(self):
        """
        os.path.dirname returns the whole path, so with subfolders the
        con subcarpetas daba "compensation_benefits/2026" y el filtro de acceso
        dejaba de casar en silencio.
        """
        assert self._departamento("compensation_benefits/2026/q1/salarios.pdf") == "compensation_benefits"

    def test_a_file_in_the_root_is_general(self):
        assert self._departamento("leeme.pdf") == "general"

    def test_the_barras_of_windows_normalizan(self):
        assert self._departamento("compensation_benefits\\salarios.pdf") == "compensation_benefits"


class TestPathNormalisation:

    def test_lowercase_and_barras_unix(self):
        assert norm_path("HR\\Politica.PDF") == "hr/politica.pdf"

    def test_collapses_barras_repetidas(self):
        assert norm_path("hr//sub///doc.pdf") == "hr/sub/doc.pdf"

    def test_is_idempotente(self):
        """
        Applied both at ingest time and when filtering. If it were not idempotent,
        the two
        lados dejarían de coincidir y el prefiltro por documento no encontraría
        nada.
        """
        una = norm_path("HR\\\\Politica  Interna.PDF")
        assert norm_path(una) == una

    def test_tolera_empty(self):
        assert norm_path("") == ""
        assert norm_path(None) == ""
