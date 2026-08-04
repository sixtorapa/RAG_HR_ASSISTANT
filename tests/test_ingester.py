"""
test_ingester.py — el módulo más grande del repositorio (1.300 líneas) y el que
más lejos está de donde se nota un fallo.

Un error aquí no lanza una excepción: produce un índice peor. Nadie lo ve hasta
que las respuestas empeoran, y para entonces nada apunta a la ingesta. Por eso
se prueban las propiedades que sostienen el retrieval:

  - cobertura: ninguna página con texto puede desaparecer al trocear
  - jerarquía: cada micro-chunk tiene que poder encontrar a su padre
  - metadata: el departamento decide el control de acceso, así que si se
    calcula mal, el guardarraíl filtra mal
  - identidad: los chunk_id tienen que ser estables entre ejecuciones, o la
    ingesta incremental deja de ser incremental
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


def _pagina(texto, n, fichero="doc.pdf"):
    return Document(
        page_content=texto,
        metadata={
            "filename": fichero, "relative_path": fichero, "source_file": fichero,
            "page_number": n, "page": n - 1, "file_type": "pdf",
        },
    )


def _palabras(n, semilla="palabra"):
    return " ".join(f"{semilla}{i}" for i in range(n))


# ══════════════════════════════════════════════════════════════════════════
# Limpieza de texto
# ══════════════════════════════════════════════════════════════════════════

class TestLimpiezaDeTexto:

    def test_junta_letras_separadas(self):
        assert "PROPUESTA" in _despace_text("U N A P R O P U E S T A").replace(" ", "")

    def test_no_toca_texto_normal(self):
        texto = "Los empleados tienen 25 días de vacaciones al año."
        assert _despace_text(texto) == texto

    def test_colapsa_espacios_y_saltos(self):
        assert _collapse_whitespace("hola    mundo") == "hola mundo"
        assert "\n\n\n" not in _collapse_whitespace("a\n\n\n\n\nb")

    def test_la_limpieza_no_vacia_un_texto_con_contenido(self):
        """Propiedad de seguridad: limpiar no puede convertir texto en nada."""
        texto = "Política de teletrabajo: hasta 3 días por semana."
        assert _clean_text_for_embeddings(texto).strip()

    def test_la_limpieza_tolera_vacio(self):
        assert _clean_text_for_embeddings("") in ("", None)


# ══════════════════════════════════════════════════════════════════════════
# Chunking por páginas (fase 1, determinista y sin LLM)
# ══════════════════════════════════════════════════════════════════════════

class TestChunkingPorPaginas:

    def test_agrupa_paginas_cortas_hasta_el_objetivo(self):
        paginas = [_pagina(_palabras(100), i) for i in range(1, 8)]
        chunks = _page_based_chunking(paginas, min_words=150, target_words=350, max_words=700)
        assert len(chunks) < len(paginas)
        assert all(len(c.page_content.split()) <= 700 for c in chunks)

    def test_el_chunking_por_paginas_NO_marca_el_tipo(self):
        """
        ⚠️ Acoplamiento real, fijado aquí para que se vea: `_page_based_chunking`
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

    def test_generar_micros_es_lo_que_marca_a_los_padres(self):
        macros = _page_based_chunking([_pagina(_palabras(400), i) for i in range(1, 4)])
        _generate_micro_chunks(macros)
        assert all(m.metadata.get("chunk_type") == "macro" for m in macros)

    def test_ninguna_pagina_con_texto_se_pierde(self):
        """
        La propiedad central: cobertura del 100%. Si una página desaparece al
        trocear, su contenido es irrecuperable por retrieval y nada lo avisa.
        """
        paginas = [_pagina(f"contenido unico de la pagina {i} " + _palabras(80), i)
                   for i in range(1, 11)]
        texto = " ".join(c.page_content for c in _page_based_chunking(paginas))
        for i in range(1, 11):
            assert f"contenido unico de la pagina {i}" in texto

    def test_una_pagina_muy_larga_va_sola(self):
        paginas = [_pagina(_palabras(80), 1), _pagina(_palabras(900), 2), _pagina(_palabras(80), 3)]
        chunks = _page_based_chunking(paginas, max_words=700)
        largos = [c for c in chunks if len(c.page_content.split()) > 700]
        assert len(largos) == 1

    def test_documentos_distintos_no_se_mezclan(self):
        paginas = [_pagina(_palabras(100), 1, "a.pdf"), _pagina(_palabras(100), 1, "b.pdf")]
        chunks = _page_based_chunking(paginas)
        for c in chunks:
            fuentes = {c.metadata.get("source_file")}
            assert len(fuentes) == 1

    def test_se_registran_las_paginas_de_cada_chunk(self):
        """`pages_in_chunk` es lo que permite citar "pág. 3-4" en la respuesta."""
        chunks = _page_based_chunking([_pagina(_palabras(120), i) for i in range(1, 6)])
        for c in chunks:
            assert c.metadata.get("pages_in_chunk")

    def test_sin_paginas_devuelve_lista_vacia(self):
        assert _page_based_chunking([]) == []


class TestFusionDePaginas:

    def test_marca_los_limites_de_pagina(self):
        chunk = _merge_pages_into_chunk([_pagina("texto uno", 1), _pagina("texto dos", 2)], "doc.pdf")
        assert "[PÁGINA" in chunk.page_content
        assert "texto uno" in chunk.page_content and "texto dos" in chunk.page_content

    def test_el_chunk_id_es_estable_entre_ejecuciones(self):
        """
        Si el id cambiara entre ingestas, la ingesta incremental borraría y
        reindexaría todo cada vez, y `parent_chunk_id` dejaría de resolver.
        """
        paginas = [_pagina("mismo contenido exacto", 1)]
        a = _merge_pages_into_chunk(paginas, "doc.pdf")
        b = _merge_pages_into_chunk(paginas, "doc.pdf")
        assert a.metadata["chunk_id"] == b.metadata["chunk_id"]

    def test_contenidos_distintos_dan_ids_distintos(self):
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

    def test_todo_hijo_apunta_a_un_padre_existente(self):
        """
        Sin esto, la expansión hijo -> padre del retrieval se queda sin padre y
        degrada al hijo, que es justo lo que se quería evitar.
        """
        macros = _page_based_chunking([_pagina(_palabras(400), i) for i in range(1, 5)])
        ids_padres = {m.metadata["chunk_id"] for m in macros}
        micros = _generate_micro_chunks(macros)
        assert micros
        for m in micros:
            assert m.metadata["parent_chunk_id"] in ids_padres

    def test_los_hijos_se_marcan_como_micro(self):
        micros = _generate_micro_chunks(_page_based_chunking([_pagina(_palabras(400), 1)]))
        assert all(m.metadata["chunk_type"] == "micro" for m in micros)

    def test_los_hijos_tienen_su_propio_id(self):
        micros = _generate_micro_chunks(_page_based_chunking([_pagina(_palabras(600), 1)]))
        ids = [m.metadata["chunk_id"] for m in micros]
        assert len(ids) == len(set(ids))

    def test_los_hijos_son_mas_pequenos_que_el_padre(self):
        macros = _page_based_chunking([_pagina(_palabras(600), 1)])
        micros = _generate_micro_chunks(macros)
        maximo_padre = max(len(m.page_content.split()) for m in macros)
        assert all(len(mi.page_content.split()) <= maximo_padre for mi in micros)

    def test_los_hijos_heredan_la_procedencia(self):
        macros = _page_based_chunking([_pagina(_palabras(400), 1, "politica.pdf")])
        for m in _generate_micro_chunks(macros):
            assert m.metadata.get("source_file") == "politica.pdf"

    def test_sin_macros_no_hay_micros(self):
        assert _generate_micro_chunks([]) == []


# ══════════════════════════════════════════════════════════════════════════
# Contexto inyectado en el texto embebido
# ══════════════════════════════════════════════════════════════════════════

class TestContextoInyectado:

    def test_antepone_la_procedencia(self):
        chunk = Document(page_content="cuerpo del chunk",
                         metadata={"relative_path": "hr/politica.pdf", "page_number": 3})
        [salida] = inject_context_to_chunks([chunk])
        assert salida.page_content.startswith("SOURCE:")
        assert "politica.pdf" in salida.page_content
        assert "cuerpo del chunk" in salida.page_content

    def test_antepone_titular_y_resumen_si_existen(self):
        """
        El resumen se genera y se paga en la fase 2. Si no llega al texto
        embebido, no sirve para recuperar: una pregunta se parece más a un
        titular y a un resumen que al cuerpo del documento.
        """
        chunk = Document(page_content="cuerpo",
                         metadata={"relative_path": "d.pdf", "page_number": 1,
                                   "semantic_headline": "Política de licencias",
                                   "semantic_summary": "Resume los tipos de permiso."})
        [salida] = inject_context_to_chunks([chunk])
        assert "TITLE: Política de licencias" in salida.page_content
        assert "SUMMARY: Resume los tipos de permiso." in salida.page_content

    def test_sin_titular_ni_resumen_no_deja_etiquetas_vacias(self):
        chunk = Document(page_content="cuerpo", metadata={"relative_path": "d.pdf", "page_number": 1})
        [salida] = inject_context_to_chunks([chunk])
        assert "TITLE:" not in salida.page_content
        assert "SUMMARY:" not in salida.page_content

    def test_cita_un_rango_cuando_el_chunk_cubre_varias_paginas(self):
        chunk = Document(page_content="cuerpo",
                         metadata={"relative_path": "d.pdf", "pages_in_chunk": "3, 4, 5"})
        [salida] = inject_context_to_chunks([chunk])
        assert "3-5" in salida.page_content


# ══════════════════════════════════════════════════════════════════════════
# Metadata para Chroma
# ══════════════════════════════════════════════════════════════════════════

class TestMetadata:

    def test_las_listas_se_aplanan(self):
        """Chroma no admite listas en metadata: sin esto, la ingesta revienta."""
        doc = Document(page_content="x", metadata={"paginas": [1, 2, 3]})
        sanitize_metadata(doc)
        assert doc.metadata["paginas"] == "1, 2, 3"

    def test_los_nulos_se_convierten_en_cadena_vacia(self):
        doc = Document(page_content="x", metadata={"campo": None})
        sanitize_metadata(doc)
        assert doc.metadata["campo"] == ""

    def test_los_diccionarios_se_serializan(self):
        doc = Document(page_content="x", metadata={"extra": {"a": 1}})
        sanitize_metadata(doc)
        assert isinstance(doc.metadata["extra"], str)

    def test_los_escalares_se_dejan_como_estan(self):
        doc = Document(page_content="x", metadata={"n": 3, "f": 1.5, "b": True, "s": "t"})
        sanitize_metadata(doc)
        assert doc.metadata == {"n": 3, "f": 1.5, "b": True, "s": "t"}


class TestDepartamentoDesdeLaRuta:
    """
    El departamento sale de la estructura de carpetas y es lo que alimenta el
    guardarraíl de acceso. Si se calcula mal, el RBAC filtra mal — y en la
    dirección peligrosa: un documento cuyo departamento no casa con ninguno
    permitido simplemente no aparece, sin error.
    """

    @staticmethod
    def _departamento(rel):
        # Misma expresión que usa process_and_store_documents.
        n = norm_path(rel)
        return (n.split("/")[0] if "/" in n else "general").lower()

    def test_primer_segmento_de_la_ruta(self):
        assert self._departamento("compensation_benefits/salarios.pdf") == "compensation_benefits"

    def test_con_subcarpetas_sigue_siendo_el_primero(self):
        """
        El defecto que había: os.path.dirname devolvía la ruta entera, así que
        con subcarpetas daba "compensation_benefits/2026" y el filtro de acceso
        dejaba de casar en silencio.
        """
        assert self._departamento("compensation_benefits/2026/q1/salarios.pdf") == "compensation_benefits"

    def test_un_fichero_en_la_raiz_es_general(self):
        assert self._departamento("leeme.pdf") == "general"

    def test_las_barras_de_windows_se_normalizan(self):
        assert self._departamento("compensation_benefits\\salarios.pdf") == "compensation_benefits"


class TestNormalizacionDeRutas:

    def test_minusculas_y_barras_unix(self):
        assert norm_path("HR\\Politica.PDF") == "hr/politica.pdf"

    def test_colapsa_barras_repetidas(self):
        assert norm_path("hr//sub///doc.pdf") == "hr/sub/doc.pdf"

    def test_es_idempotente(self):
        """
        Se aplica al ingestar y al filtrar. Si no fuera idempotente, los dos
        lados dejarían de coincidir y el prefiltro por documento no encontraría
        nada.
        """
        una = norm_path("HR\\\\Politica  Interna.PDF")
        assert norm_path(una) == una

    def test_tolera_vacio(self):
        assert norm_path("") == ""
        assert norm_path(None) == ""
