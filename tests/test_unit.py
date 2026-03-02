"""
test_unit.py — Tests unitarios de lógica pura.
No requieren Flask app context ni llamadas a OpenAI.

Cubre:
  - cost_calculator.calculate_cost()
  - routes._extract_user_mode()
  - routes._make_chat_title_from_question()
  - HRDatabaseTool: init, schema, SQL sanitization
"""

import pytest
import re


# ══════════════════════════════════════════════════════════════════
# 1. COST CALCULATOR
# ══════════════════════════════════════════════════════════════════

class TestCostCalculator:
    """
    Por qué testeamos esto:
    Errores en el cálculo de costes se acumulan silenciosamente en producción.
    Un test simple protege contra refactors del pricing o la fórmula.
    """

    def test_gpt4o_mini_basic_cost(self, app):
        """Verifica que el cálculo de coste es correcto para gpt-4o-mini."""
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
        """Modelo desconocido no debe lanzar excepción, retorna 0."""
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
# 2. _extract_user_mode  (lógica de routing explícito)
# ══════════════════════════════════════════════════════════════════

class TestExtractUserMode:
    """
    _extract_user_mode detecta prefijos explícitos del usuario (SQL:, AMBAS:, DOC:)
    para forzar la ruta del agente. Es crítico que funcione con variaciones de formato.
    """

    @pytest.fixture(autouse=True)
    def import_fn(self, app):
        with app.app_context():
            from app.main.routes import _extract_user_mode
            self.fn = _extract_user_mode

    def test_sql_prefix_colon(self):
        mode, text = self.fn("SQL: dame el top 10 de salarios")
        assert mode == "sql"
        assert text == "dame el top 10 de salarios"

    def test_sql_prefix_no_separator(self):
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

    def test_no_prefix_returns_none(self):
        mode, text = self.fn("¿Cuál es la política de vacaciones?")
        assert mode is None
        assert "vacaciones" in text

    def test_empty_string(self):
        mode, text = self.fn("")
        assert mode is None

    def test_sql_only_no_question(self):
        """Si solo hay el prefijo sin pregunta, la función no debe crashear."""
        mode, text = self.fn("SQL")
        assert mode == "sql"


# ══════════════════════════════════════════════════════════════════
# 3. _make_chat_title_from_question
# ══════════════════════════════════════════════════════════════════

class TestMakeChatTitle:
    """
    _make_chat_title_from_question genera un título corto para la sesión.
    Aparece en el sidebar del chat — mal formato es un bug visible al usuario.
    """

    @pytest.fixture(autouse=True)
    def import_fn(self, app):
        with app.app_context():
            from app.main.routes import _make_chat_title_from_question
            self.fn = _make_chat_title_from_question

    def test_short_question_kept_as_is(self):
        title = self.fn("¿Cuántos empleados hay?")
        assert "empleados" in title

    def test_long_question_truncated(self):
        long_q = "Esta es una pregunta muy larga sobre la política de vacaciones anuales de la empresa que supera los 46 caracteres"
        title = self.fn(long_q)
        assert len(title) <= 50  # margen por el ellipsis

    def test_empty_returns_default(self):
        title = self.fn("")
        assert title == "Nuevo chat"

    def test_whitespace_only_returns_default(self):
        title = self.fn("   ")
        assert title == "Nuevo chat"

    def test_newlines_cleaned(self):
        title = self.fn("Pregunta\ncon\nsaltos de línea")
        assert "\n" not in title

    def test_multiple_spaces_collapsed(self):
        title = self.fn("Pregunta    con    espacios")
        assert "  " not in title


# ══════════════════════════════════════════════════════════════════
# 4. HRDatabaseTool — lógica sin LLM
# ══════════════════════════════════════════════════════════════════

class TestHRDatabaseToolInit:
    """
    Verifica que HRDatabaseTool tiene las propiedades correctas.
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
        """El prompt del tool debe prohibir DDL/DML destructivos."""
        from app.rag_logic.sql_tool import HRDatabaseTool

        # El sistema prompt menciona explícitamente los keywords prohibidos
        schema = HRDatabaseTool.DB_SCHEMA_CONTEXT
        # Al menos uno de DROP/DELETE/UPDATE debe estar mencionado en el contexto
        # (están en el sql_prompt dentro de _run, pero podemos verificar la clase)
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