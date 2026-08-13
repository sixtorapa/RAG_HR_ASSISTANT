"""
test_loaders.py — PDF loading, and in particular the table de-duplicator.

`_strip_table_lines` DELETES text from the corpus. That is the kind of code to
protect before any other: a mistake here raises no error, breaks no integration
test, and is not noticed until the answers get worse weeks later, with nothing
pointing at ingestion.

The de-duplicator's criterion is deliberately conservative — only what matches
EXACTLY a full row or a long cell is removed — and these tests exist to keep it
that way: half of them assert what must NOT be removed.
"""

import pytest
from langchain.docstore.document import Document

from app.rag_logic.custom_loaders import BetterPDFLoader as L


TABLA = """| Leave Type | Entitlement | Notice Required |
| --- | --- | --- |
| Annual Leave | 25 days/year | 2 weeks minimum |
| Sick Leave | Unlimited (certified) | Same day (before 10:00) |"""


class TestWhatIsRemoved:

    def test_removes_a_row_repetida_as_text_standalone(self):
        text = "Annual Leave 25 days/year 2 weeks minimum"
        assert L._strip_table_lines(text, TABLA).strip() == ""

    def test_removes_several_rows(self):
        text = (
            "Annual Leave 25 days/year 2 weeks minimum\n"
            "Sick Leave Unlimited (certified) Same day (before 10:00)"
        )
        assert L._strip_table_lines(text, TABLA).strip() == ""

    def test_removes_the_cabecera_repetida(self):
        assert L._strip_table_lines("Leave Type Entitlement Notice Required", TABLA).strip() == ""

    def test_is_indiferente_a_uppercase_and_spaces(self):
        text = "  ANNUAL   LEAVE   25 days/year   2 weeks minimum  "
        assert L._strip_table_lines(text, TABLA).strip() == ""

    def test_removes_a_cell_larga_standalone(self):
        # >= 12 caracteres: "Unlimited (certified)"
        assert L._strip_table_lines("Unlimited (certified)", TABLA).strip() == ""

    def test_the_caso_real_of_the_indice(self):
        """The row seen duplicated in the evaluation dumps."""
        table = "| 4.5 – 5.0 | Outstanding | Top 10%. Exceptional impact and leadership. |"
        text = "4.5 – 5.0 Outstanding Top 10%. Exceptional impact and leadership."
        assert L._strip_table_lines(text, table).strip() == ""


class TestWhatMustNotBeRemoved:
    """
    The important half. Over-deleting is far worse than leaving a repetition:
    el contenido perdido no vuelve, y no hay forma de notarlo.
    """

    def test_keeps_the_text_that_not_is_of_the_table(self):
        text = "3. Leave Policy\n\nAnnual Leave 25 days/year 2 weeks minimum\n\n4. Code of Conduct"
        output = L._strip_table_lines(text, TABLA)
        assert "3. Leave Policy" in output
        assert "4. Code of Conduct" in output
        assert "Annual Leave 25 days/year" not in output

    def test_not_removes_a_phrase_that_only_menciona_a_value(self):
        text = "Employees are entitled to 25 days/year of annual leave, as shown above."
        assert text in L._strip_table_lines(text, TABLA)

    def test_not_removes_cells_cortas(self):
        """
        A short cell ("25", "Sí", "N/A") may well be legitimate text
        anywhere in the document.
        """
        table = "| Días | 25 |\n| --- | --- |\n| Aviso | 2 |"
        text = "El artículo 25 regula el preaviso."
        assert text in L._strip_table_lines(text, table)

    def test_a_match_partial_not_suffices(self):
        text = "Annual Leave is granted at 25 days/year after probation"
        assert text in L._strip_table_lines(text, TABLA)

    def test_without_table_returns_the_text_untouched(self):
        text = "Cualquier text\ncon varias líneas"
        assert L._strip_table_lines(text, "") == text
        assert L._strip_table_lines(text, None) == text

    def test_without_text_not_blows_up(self):
        assert L._strip_table_lines("", TABLA) == ""
        assert L._strip_table_lines(None, TABLA) is None

    def test_ignores_the_row_separadora(self):
        """
        `| --- | --- |` must not become a deletion pattern; if it
        did, every dashed line in the document would disappear.
        """
        text = "--- --- ---\nSección siguiente"
        output = L._strip_table_lines(text, TABLA)
        assert "Sección siguiente" in output


class TestGapCollapsing:

    def test_not_lets_bloques_of_lines_in_blanco(self):
        text = "Título\n\nAnnual Leave 25 days/year 2 weeks minimum\n\nSick Leave Unlimited (certified) Same day (before 10:00)\n\nFinal"
        output = L._strip_table_lines(text, TABLA)
        assert "\n\n\n" not in output
        assert "Título" in output and "Final" in output


class TestTablesToMarkdown:

    def test_convierte_a_table_simple(self):
        md = L._tables_to_markdown([[["A", "B"], ["1", "2"]]])
        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md

    def test_the_cells_nulas_vacian(self):
        md = L._tables_to_markdown([[["A", None], ["1", "2"]]])
        assert "| A |  |" in md

    def test_line_breaks_inside_a_cell_are_flattened(self):
        md = L._tables_to_markdown([[["Leave\nType"], ["Annual"]]])
        assert "Leave Type" in md
        assert "Leave\nType" not in md

    def test_the_rows_completamente_empty_descartan(self):
        md = L._tables_to_markdown([[["A", "B"], ["", ""], ["1", "2"]]])
        assert md.count("\n|") >= 2
        assert "|  |  |" not in md

    def test_without_tables_returns_string_empty(self):
        assert L._tables_to_markdown([]) == ""
        assert L._tables_to_markdown([[]]) == ""


class TestPageDocument:

    @pytest.fixture
    def loader(self):
        return L("/ruta/falsa/documento.pdf")

    def test_the_table_comes_before_the_text(self, loader):
        doc = loader._build_page_document(
            0, 3, "Texto corrido de la página.", TABLA, "documento.pdf",
        )
        pos_tabla = doc.page_content.index("TABLE DATA")
        pos_texto = doc.page_content.index("Texto corrido")
        assert pos_tabla < pos_texto

    def test_the_deduplicador_aplica_at_construir_the_page(self, loader):
        """The integration that matters: the row does not travel twice in the chunk."""
        doc = loader._build_page_document(
            0, 3,
            "Leave Policy\nAnnual Leave 25 days/year 2 weeks minimum",
            TABLA, "documento.pdf",
        )
        # Exactly once, inside the markdown block
        assert doc.page_content.count("25 days/year") == 1
        assert "Leave Policy" in doc.page_content

    def test_a_page_without_text_marks_as_empty(self, loader):
        doc = loader._build_page_document(4, 10, "", "", "documento.pdf")
        assert doc.metadata["is_empty_page"] is True
        assert "Page 5" in doc.page_content

    def test_the_metadata_of_page_is_coherente(self, loader):
        doc = loader._build_page_document(2, 10, "contenido suficiente aquí", "", "documento.pdf")
        assert doc.metadata["page"] == 2
        assert doc.metadata["page_number"] == 3      # 1-indexado para citar
        assert doc.metadata["page_count"] == 10
        assert doc.metadata["is_empty_page"] is False

    def test_marks_whether_the_page_has_a_table(self, loader):
        con = loader._build_page_document(0, 1, "texto de la pagina", TABLA, "d.pdf")
        sin = loader._build_page_document(0, 1, "texto de la pagina", "", "d.pdf")
        assert con.metadata["has_table"] is True
        assert sin.metadata["has_table"] is False

    def test_returns_a_document_of_langchain(self, loader):
        doc = loader._build_page_document(0, 1, "texto", "", "d.pdf")
        assert isinstance(doc, Document)
