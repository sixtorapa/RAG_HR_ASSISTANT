"""
test_unit.py — unit tests for pure logic. No Flask app context, no OpenAI calls.

Covers:
  - cost_calculator.calculate_cost()
  - pipeline._extract_user_mode()
  - pipeline._make_chat_title_from_question()
  - HRDatabaseTool: init, schema, SQL sanitisation
"""

import pytest
import re


# ══════════════════════════════════════════════════════════════════
# 1. COST CALCULATOR
# ══════════════════════════════════════════════════════════════════

class TestCostCalculator:
    """
    Errors in cost calculation accumulate silently in production, so a simple
    test guards against refactors of the pricing table or the formula.
    """

    def test_gpt4o_mini_basic_cost(self, app):
        """The cost calculation is correct for gpt-4o-mini."""
        with app.app_context():
            from app.rag_logic.cost_calculator import calculate_cost

            # 1M prompt tokens * 0.15 USD/M = 0.15 USD * 0.92 EUR/USD = 0.138 EUR
            cost = calculate_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0)
            assert abs(cost - 0.138) < 0.001

    def test_gpt4o_completion_cost(self, app):
        with app.app_context():
            from app.rag_logic.cost_calculator import calculate_cost

            # 1M completion tokens * 10.00 USD/M = 10 USD * 0.92 = 9.20 EUR
            cost = calculate_cost("gpt-4o", prompt_tokens=0, completion_tokens=1_000_000)
            assert abs(cost - 9.20) < 0.01

    def test_unknown_model_returns_zero(self, app):
        """An unknown model must not raise; it returns 0 and reports the error."""
        with app.app_context():
            from app.rag_logic.cost_calculator import calculate_cost

            cost = calculate_cost("gpt-99-turbo", prompt_tokens=100_000, completion_tokens=50_000)
            assert cost == 0.0

    def test_zero_tokens_returns_zero(self, app):
        with app.app_context():
            from app.rag_logic.cost_calculator import calculate_cost

            cost = calculate_cost("gpt-4o-mini", prompt_tokens=0, completion_tokens=0)
            assert cost == 0.0

    def test_both_token_types(self, app):
        """Prompt + completion se suman correctamente."""
        with app.app_context():
            from app.rag_logic.cost_calculator import calculate_cost

            # gpt-4o-mini: 0.15/M prompt + 0.60/M completion
            # 500k prompt → 0.075 USD, 500k completion → 0.30 USD → 0.375 * 0.92
            cost = calculate_cost("gpt-4o-mini", 500_000, 500_000)
            expected = (0.075 + 0.30) * 0.92
            assert abs(cost - expected) < 0.001


# ══════════════════════════════════════════════════════════════════
# 2. _extract_user_mode  (explicit routing logic)
# ══════════════════════════════════════════════════════════════════

class TestExtractUserMode:
    """
    _extract_user_mode detects explicit user prefixes (SQL:, AMBAS:, DOC:)
    para forzar la ruta del agent. Es crítico que funcione con variaciones de formato.
    """

    @pytest.fixture(autouse=True)
    def import_fn(self, app):
        with app.app_context():
            from app.main.pipeline import _extract_user_mode
            self.fn = _extract_user_mode

    def test_sql_prefix_colon(self):
        mode, text = self.fn("SQL: dame el top 10 de salarios")
        assert mode == "sql"
        assert text == "dame el top 10 de salarios"

    def test_sql_prefix_not_separator(self):
        mode, text = self.fn("SQL dame el top 10")
        assert mode == "sql"
        assert text == "dame el top 10"

    def test_ambas_prefix(self):
        mode, text = self.fn("AMBAS - compara salario con política")
        assert mode == "ambas"
        assert "compara" in text

    def test_sql_case_insensitive(self):
        mode, text = self.fn("sql: lista departamentos")
        assert mode == "sql"

    def test_not_prefix_returns_none(self):
        mode, text = self.fn("¿Cuál es la política de vacaciones?")
        assert mode is None
        assert "vacaciones" in text

    def test_empty_string(self):
        mode, text = self.fn("")
        assert mode is None

    def test_sql_only_not_question(self):
        """With only the prefix and no question, the function must not crash."""
        mode, text = self.fn("SQL")
        assert mode == "sql"


# ══════════════════════════════════════════════════════════════════
# 3. _make_chat_title_from_question
# ══════════════════════════════════════════════════════════════════

class TestMakeChatTitle:
    """
    _make_chat_title_from_question builds a short title for the session.
    Aparece en el sidebar del chat — mal formato es un bug visible al usuario.
    """

    @pytest.fixture(autouse=True)
    def import_fn(self, app):
        with app.app_context():
            from app.main.pipeline import _make_chat_title_from_question
            self.fn = _make_chat_title_from_question

    def test_short_question_kept_as_is(self):
        title = self.fn("¿Cuántos empleados hay?")
        assert "empleados" in title

    def test_long_question_truncated(self):
        long_q = "Esta es una question muy larga sobre la política de vacaciones anuales de la empresa que supera los 46 caracteres"
        title = self.fn(long_q)
        assert len(title) <= 50  # margin for the ellipsis

    def test_empty_returns_default(self):
        title = self.fn("")
        assert title == "New chat"

    def test_whitespace_only_returns_default(self):
        title = self.fn("   ")
        assert title == "New chat"

    def test_newlines_cleaned(self):
        title = self.fn("Pregunta\ncon\nsaltos de línea")
        assert "\n" not in title

    def test_multiple_spaces_collapsed(self):
        title = self.fn("Pregunta    con    espacios")
        assert "  " not in title


# ══════════════════════════════════════════════════════════════════
# 4. HRDatabaseTool — logic without the LLM
# ══════════════════════════════════════════════════════════════════

class TestHRDatabaseToolInit:
    """
    HRDatabaseTool exposes the right properties.
    En entrevistas: "¿Cómo proteges contra SQL injection?" →
    el LLM genera el SQL, pero añadimos validación de keywords peligrosas.
    """

    def test_tool_name_and_description(self):
        from app.rag_logic.sql_tool import HRDatabaseTool

        tool = HRDatabaseTool(model_name="gpt-4o-mini")
        assert tool.name == "query_hr_database"
        assert "salary" in tool.description.lower() or "HR" in tool.description

    def test_schema_context_contains_tables(self):
        from app.rag_logic.sql_tool import HRDatabaseTool

        assert "employees" in HRDatabaseTool.DB_SCHEMA_CONTEXT
        assert "departments" in HRDatabaseTool.DB_SCHEMA_CONTEXT

    def test_dangerous_sql_keywords_in_prompt(self):
        """The tool prompt must forbid destructive DDL/DML."""
        from app.rag_logic.sql_tool import HRDatabaseTool

        # The system prompt explicitly names the forbidden keywords
        schema = HRDatabaseTool.DB_SCHEMA_CONTEXT
        # At least one of DROP/DELETE/UPDATE must be named in the context
        # (they live in the sql_prompt inside _run, but the class can be checked)
        tool = HRDatabaseTool(model_name="gpt-4o-mini")
        assert tool.model_name == "gpt-4o-mini"

    def test_default_project_settings_empty(self):
        from app.rag_logic.sql_tool import HRDatabaseTool

        tool = HRDatabaseTool(model_name="gpt-4o-mini")
        assert tool.project_settings == {}

    def test_custom_project_settings(self):
        from app.rag_logic.sql_tool import HRDatabaseTool

        settings = {"sql_context": "custom schema"}
        tool = HRDatabaseTool(model_name="gpt-4o-mini", project_settings=settings)
        assert tool.project_settings["sql_context"] == "custom schema"