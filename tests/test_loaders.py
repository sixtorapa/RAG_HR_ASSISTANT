"""
test_loaders.py — la carga de PDF, y en particular el deduplicador de tablas.

`_strip_table_lines` BORRA texto del corpus. Es la clase de código que hay que
blindar antes que ninguna otra: un fallo aquí no da error, no rompe ningún
test de integración y no se nota hasta que las respuestas empeoran semanas
después, sin que nada apunte a la ingesta.

El criterio del deduplicador es deliberadamente conservador —solo se borra lo
que coincide EXACTAMENTE con una fila o una celda larga— y estos tests existen
para que siga siéndolo: la mitad de ellos comprueban lo que NO se debe borrar.
"""

import pytest
from langchain.docstore.document import Document

from app.rag_logic.custom_loaders import BetterPDFLoader as L


TABLA = """| Leave Type | Entitlement | Notice Required |
| --- | --- | --- |
| Annual Leave | 25 days/year | 2 weeks minimum |
| Sick Leave | Unlimited (certified) | Same day (before 10:00) |"""


class TestLoQueSeBorra:

    def test_borra_una_fila_repetida_como_texto_suelto(self):
        texto = "Annual Leave 25 days/year 2 weeks minimum"
        assert L._strip_table_lines(texto, TABLA).strip() == ""

    def test_borra_varias_filas(self):
        texto = (
            "Annual Leave 25 days/year 2 weeks minimum\n"
            "Sick Leave Unlimited (certified) Same day (before 10:00)"
        )
        assert L._strip_table_lines(texto, TABLA).strip() == ""

    def test_borra_la_cabecera_repetida(self):
        assert L._strip_table_lines("Leave Type Entitlement Notice Required", TABLA).strip() == ""

    def test_es_indiferente_a_mayusculas_y_espacios(self):
        texto = "  ANNUAL   LEAVE   25 days/year   2 weeks minimum  "
        assert L._strip_table_lines(texto, TABLA).strip() == ""

    def test_borra_una_celda_larga_suelta(self):
        # >= 12 caracteres: "Unlimited (certified)"
        assert L._strip_table_lines("Unlimited (certified)", TABLA).strip() == ""

    def test_el_caso_real_del_indice(self):
        """La fila que se vio duplicada en los volcados de evaluación."""
        tabla = "| 4.5 – 5.0 | Outstanding | Top 10%. Exceptional impact and leadership. |"
        texto = "4.5 – 5.0 Outstanding Top 10%. Exceptional impact and leadership."
        assert L._strip_table_lines(texto, tabla).strip() == ""


class TestLoQueNOSeDebeBorrar:
    """
    La mitad importante. Borrar de más es mucho peor que dejar una repetición:
    el contenido perdido no vuelve, y no hay forma de notarlo.
    """

    def test_conserva_el_texto_que_no_es_de_la_tabla(self):
        texto = "3. Leave Policy\n\nAnnual Leave 25 days/year 2 weeks minimum\n\n4. Code of Conduct"
        salida = L._strip_table_lines(texto, TABLA)
        assert "3. Leave Policy" in salida
        assert "4. Code of Conduct" in salida
        assert "Annual Leave 25 days/year" not in salida

    def test_no_borra_una_frase_que_solo_menciona_un_valor(self):
        texto = "Employees are entitled to 25 days/year of annual leave, as shown above."
        assert texto in L._strip_table_lines(texto, TABLA)

    def test_no_borra_celdas_cortas(self):
        """
        Una celda de pocos caracteres ("25", "Sí", "N/A") puede ser texto
        legítimo en cualquier parte del documento.
        """
        tabla = "| Días | 25 |\n| --- | --- |\n| Aviso | 2 |"
        texto = "El artículo 25 regula el preaviso."
        assert texto in L._strip_table_lines(texto, tabla)

    def test_una_coincidencia_parcial_no_basta(self):
        texto = "Annual Leave is granted at 25 days/year after probation"
        assert texto in L._strip_table_lines(texto, TABLA)

    def test_sin_tabla_devuelve_el_texto_intacto(self):
        texto = "Cualquier texto\ncon varias líneas"
        assert L._strip_table_lines(texto, "") == texto
        assert L._strip_table_lines(texto, None) == texto

    def test_sin_texto_no_revienta(self):
        assert L._strip_table_lines("", TABLA) == ""
        assert L._strip_table_lines(None, TABLA) is None

    def test_ignora_la_fila_separadora(self):
        """
        `| --- | --- |` no debe convertirse en un patrón a borrar; si lo
        hiciera, cualquier línea de guiones del documento desaparecería.
        """
        texto = "--- --- ---\nSección siguiente"
        salida = L._strip_table_lines(texto, TABLA)
        assert "Sección siguiente" in salida


class TestColapsoDeHuecos:

    def test_no_deja_bloques_de_lineas_en_blanco(self):
        texto = "Título\n\nAnnual Leave 25 days/year 2 weeks minimum\n\nSick Leave Unlimited (certified) Same day (before 10:00)\n\nFinal"
        salida = L._strip_table_lines(texto, TABLA)
        assert "\n\n\n" not in salida
        assert "Título" in salida and "Final" in salida


class TestTablasAMarkdown:

    def test_convierte_una_tabla_simple(self):
        md = L._tables_to_markdown([[["A", "B"], ["1", "2"]]])
        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md

    def test_las_celdas_nulas_se_vacian(self):
        md = L._tables_to_markdown([[["A", None], ["1", "2"]]])
        assert "| A |  |" in md

    def test_los_saltos_dentro_de_una_celda_se_aplanan(self):
        md = L._tables_to_markdown([[["Leave\nType"], ["Annual"]]])
        assert "Leave Type" in md
        assert "Leave\nType" not in md

    def test_las_filas_completamente_vacias_se_descartan(self):
        md = L._tables_to_markdown([[["A", "B"], ["", ""], ["1", "2"]]])
        assert md.count("\n|") >= 2
        assert "|  |  |" not in md

    def test_sin_tablas_devuelve_cadena_vacia(self):
        assert L._tables_to_markdown([]) == ""
        assert L._tables_to_markdown([[]]) == ""


class TestDocumentoDePagina:

    @pytest.fixture
    def loader(self):
        return L("/ruta/falsa/documento.pdf")

    def test_la_tabla_va_delante_del_texto(self, loader):
        doc = loader._build_page_document(
            0, 3, "Texto corrido de la página.", TABLA, "documento.pdf",
        )
        pos_tabla = doc.page_content.index("DATOS TABULARES")
        pos_texto = doc.page_content.index("Texto corrido")
        assert pos_tabla < pos_texto

    def test_el_deduplicador_se_aplica_al_construir_la_pagina(self, loader):
        """La integración que importa: la fila no viaja dos veces en el chunk."""
        doc = loader._build_page_document(
            0, 3,
            "Leave Policy\nAnnual Leave 25 days/year 2 weeks minimum",
            TABLA, "documento.pdf",
        )
        # Una sola vez, dentro del bloque markdown
        assert doc.page_content.count("25 days/year") == 1
        assert "Leave Policy" in doc.page_content

    def test_una_pagina_sin_texto_se_marca_como_vacia(self, loader):
        doc = loader._build_page_document(4, 10, "", "", "documento.pdf")
        assert doc.metadata["is_empty_page"] is True
        assert "Página 5" in doc.page_content

    def test_la_metadata_de_pagina_es_coherente(self, loader):
        doc = loader._build_page_document(2, 10, "contenido suficiente aquí", "", "documento.pdf")
        assert doc.metadata["page"] == 2
        assert doc.metadata["page_number"] == 3      # 1-indexado para citar
        assert doc.metadata["page_count"] == 10
        assert doc.metadata["is_empty_page"] is False

    def test_marca_si_la_pagina_tiene_tabla(self, loader):
        con = loader._build_page_document(0, 1, "texto de la pagina", TABLA, "d.pdf")
        sin = loader._build_page_document(0, 1, "texto de la pagina", "", "d.pdf")
        assert con.metadata["has_table"] is True
        assert sin.metadata["has_table"] is False

    def test_devuelve_un_document_de_langchain(self, loader):
        doc = loader._build_page_document(0, 1, "texto", "", "d.pdf")
        assert isinstance(doc, Document)
