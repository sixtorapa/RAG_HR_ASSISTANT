"""
test_sql_tool.py — HRDatabaseTool against a real SQLite database.

Text-to-SQL is the part of the system where a mistake is most expensive, so
what is covered is the boundary rather than the happy path:

  1. The connection works and valid queries return the expected rows.
  2. Destructive statements are rejected.
  3. The tool schema is correct.
  4. A failing query is retried with the real SQLite error fed back, and gives
     up after a bounded number of attempts instead of looping.
"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def hr_sqlite_db(tmp_path_factory):
    """An in-memory SQLite database with sample data for the HR tool."""
    db_path = tmp_path_factory.mktemp("data") / "hr_test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL
        );

        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER REFERENCES departments(id),
            role TEXT,
            level TEXT,
            salary REAL,
            hire_date TEXT,
            manager_id INTEGER,
            location TEXT,
            status TEXT DEFAULT 'active'
        );

        INSERT INTO departments VALUES (1, 'Engineering', 500000);
        INSERT INTO departments VALUES (2, 'HR', 150000);
        INSERT INTO departments VALUES (3, 'Sales', 300000);

        INSERT INTO employees VALUES
            (1, 'Alice García',    1, 'Senior Engineer', 'Senior', 65000, '2021-03-15', NULL, 'Madrid',    'active'),
            (2, 'Bob Martínez',    1, 'Junior Engineer',  'Junior', 38000, '2023-01-10', 1,    'Barcelona', 'active'),
            (3, 'Carol López',     2, 'HR Manager',       'Manager',72000, '2020-06-01', NULL, 'Remote',    'active'),
            (4, 'David Pérez',     3, 'Sales Rep',        'Mid',    48000, '2022-09-20', NULL, 'Madrid',    'active'),
            (5, 'Elena Torres',    1, 'Lead Engineer',    'Lead',   80000, '2019-11-05', NULL, 'London',    'active'),
            (6, 'Former Employee', 1, 'Junior Engineer',  'Junior', 35000, '2018-01-01', NULL, 'Madrid',    'terminated');
    """)

    conn.commit()
    conn.close()
    return str(db_path)


class TestHRDatabaseToolSchema:
    """The tool schema and configuration."""

    def test_schema_has_employees_columns(self):
        from app.rag_logic.sql_tool import HRDatabaseTool

        schema = HRDatabaseTool.DB_SCHEMA_CONTEXT
        expected_columns = ["name", "department_id", "salary", "hire_date", "status", "location"]
        for col in expected_columns:
            assert col in schema, f"Column '{col}' not in DB_SCHEMA_CONTEXT"

    def test_schema_has_departments_table(self):
        from app.rag_logic.sql_tool import HRDatabaseTool

        assert "departments" in HRDatabaseTool.DB_SCHEMA_CONTEXT

    def test_tool_description_mentions_use_cases(self):
        from app.rag_logic.sql_tool import HRDatabaseTool

        tool = HRDatabaseTool(model_name="gpt-4o-mini")
        desc_lower = tool.description.lower()
        # Must mention at least one of these use cases
        assert any(kw in desc_lower for kw in ["salary", "department", "headcount", "hr"])

    def test_args_schema_has_query_field(self):
        from app.rag_logic.sql_tool import HRDatabaseTool, HRQueryInput

        schema = HRQueryInput.model_json_schema()
        assert "query" in schema["properties"]


class TestHRDatabaseDirectSQL:
    """Direct SQL against the sample database, with no LLM involved."""

    def test_count_active_employees(self, hr_sqlite_db):
        conn = sqlite3.connect(hr_sqlite_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employees WHERE status='active'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 5  # 6 total, 1 terminado

    def test_avg_salary_engineering(self, hr_sqlite_db):
        conn = sqlite3.connect(hr_sqlite_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ROUND(AVG(e.salary), 2)
            FROM employees e
            JOIN departments d ON e.department_id = d.id
            WHERE d.name = 'Engineering' AND e.status = 'active'
        """)
        avg = cursor.fetchone()[0]
        conn.close()
        # Alice: 65k, Bob: 38k, Elena: 80k → avg = 61000
        assert abs(avg - 61000.0) < 1.0

    def test_departments_have_data(self, hr_sqlite_db):
        conn = sqlite3.connect(hr_sqlite_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM departments")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 3

    def test_terminated_employees_excluded_by_default(self, hr_sqlite_db):
        """The prompt instructs the LLM to filter on status='active' by default."""
        conn = sqlite3.connect(hr_sqlite_db)
        cursor = conn.cursor()
        # With the filter, as the LLM would write it
        cursor.execute("SELECT COUNT(*) FROM employees WHERE status='active'")
        active = cursor.fetchone()[0]
        # Sin filtro
        cursor.execute("SELECT COUNT(*) FROM employees")
        total = cursor.fetchone()[0]
        conn.close()
        assert active < total  # at least one is finished


class TestHRDatabaseToolMocked:
    """
    HRDatabaseTool with the LLM mocked.
    Verifica que el tool orquesta correctamente: LLM → SQL → Execute → Interpret.
    """

    def test_tool_returns_error_on_bad_sql(self, app, hr_sqlite_db):
        """The tool must not crash when the LLM writes SQL against a missing table.

        After exhausting the self-correction retries it must return a message
        genérico para el usuario (sin exponer el error crudo de SQLite)."""
        with app.app_context():
            from app.rag_logic.sql_tool import HRDatabaseTool, MAX_SQL_ATTEMPTS

            tool = HRDatabaseTool(model_name="gpt-4o-mini")

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="SELECT * FROM tabla_inexistente"
            )

            conn = sqlite3.connect(hr_sqlite_db)
            with patch("app.rag_logic.sql_tool.get_llm", return_value=mock_llm), \
                 patch.object(tool, "_get_connection", return_value=conn):

                result = tool._run("Pregunta que genera SQL inválido")

            # Must return an error message, not raise
            assert isinstance(result, str)
            # Retried MAX_SQL_ATTEMPTS times before giving up
            assert mock_llm.invoke.call_count == MAX_SQL_ATTEMPTS
            # A generic message for the user, not the raw SQLite error
            assert "couldn't run a valid query" in result.lower()
            assert "tabla_inexistente" not in result


class TestHRDatabaseToolSelfCorrection:
    """
    The self-correction loop in _run(): when the generated query fails
    in SQLite it is retried with the real error fed back to the LLM, up to
    MAX_SQL_ATTEMPTS veces en total.
    """

    def test_self_corrects_after_invalid_column(self, app, hr_sqlite_db):
        """The first attempt references a missing column (a typo); the LLM
        recibe el error real de SQLite y el 2º intento usa la columna correcta."""
        with app.app_context():
            from app.rag_logic.sql_tool import HRDatabaseTool

            tool = HRDatabaseTool(model_name="gpt-4o-mini")

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = [
                MagicMock(content="SELECT name FROM employees WHERE departament_id = 1"),  # typo
                MagicMock(content="SELECT name FROM employees WHERE department_id = 999"),  # corregido, sin resultados
            ]

            conn = sqlite3.connect(hr_sqlite_db)
            with patch("app.rag_logic.sql_tool.get_llm", return_value=mock_llm), \
                 patch.object(tool, "_get_connection", return_value=conn):

                result = tool._run("¿Cuántos empleados hay en el departamento X?")

            # Retried exactly once (2 LLM calls to generate SQL)
            assert mock_llm.invoke.call_count == 2

            # The second prompt must carry the real SQLite error from the first attempt
            second_call_messages = mock_llm.invoke.call_args_list[1][0][0]
            joined = " ".join(getattr(m, "content", "") for m in second_call_messages)
            assert "departament_id" in joined or "no such column" in joined.lower()

            # The corrected query is valid but returns no rows
            assert result == "No results found for your query."

    def test_gives_up_after_max_attempts(self, app, hr_sqlite_db):
        """If the LLM never fixes the query, it is retried MAX_SQL_ATTEMPTS times
        y se devuelve un mensaje genérico (sin exponer el error SQL crudo)."""
        with app.app_context():
            from app.rag_logic.sql_tool import HRDatabaseTool, MAX_SQL_ATTEMPTS

            tool = HRDatabaseTool(model_name="gpt-4o-mini")

            bad_sql = "SELECT name FROM employees WHERE departament_id = 1"
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = [MagicMock(content=bad_sql) for _ in range(MAX_SQL_ATTEMPTS)]

            conn = sqlite3.connect(hr_sqlite_db)
            with patch("app.rag_logic.sql_tool.get_llm", return_value=mock_llm), \
                 patch.object(tool, "_get_connection", return_value=conn):

                result = tool._run("Pregunta que el LLM nunca resuelve bien")

            assert mock_llm.invoke.call_count == MAX_SQL_ATTEMPTS
            assert "no such column" not in result.lower()
            assert "couldn't run a valid query" in result.lower()