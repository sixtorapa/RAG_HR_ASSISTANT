# sql_tool.py
"""
HR Analytics SQL Tool — backed by SQLite (toy data for demo / dev).
In production, swap HR_DB_URI for a real Postgres / Redshift URI via env var.
The rest of the code is identical — SQLAlchemy handles the dialect.
"""

import sqlite3
import os
import re
from typing import Optional
from urllib.parse import quote

from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from typing import ClassVar
from flask import current_app


# Intento inicial + reintentos con feedback del error real de SQLite
MAX_SQL_ATTEMPTS = 3


# ── Input schema ─────────────────────────────────────────────────────────────

class HRQueryInput(BaseModel):
    query: str = Field(description="Natural-language question about HR data.")


# ── SQL safety: defense-in-depth against LLM-generated SQL ────────────────────

_FORBIDDEN_SQL_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "REPLACE", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
)


def _is_select_only(sql: str) -> bool:
    """True if `sql` is a single read-only SELECT/CTE statement."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned or ";" in cleaned:
        return False

    first_word = cleaned.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return False

    upper = cleaned.upper()
    return not any(re.search(rf"\b{kw}\b", upper) for kw in _FORBIDDEN_SQL_KEYWORDS)


# ── Tool ─────────────────────────────────────────────────────────────────────

class HRDatabaseTool(BaseTool):
    name: str = "query_hr_database"
    description: str = (
        "Use this tool to answer questions about structured HR data: "
        "headcount, salaries, departments, performance scores, attrition, tenure. "
        "Useful for any question involving numbers, trends, rankings or comparisons."
    )
    args_schema: type[BaseModel] = HRQueryInput

    model_name: str
    project_settings: dict = {}

    # ── DB context (sent to the LLM so it can write correct SQL) ─────────────
    DB_SCHEMA_CONTEXT: ClassVar[str] = """
AVAILABLE TABLES (SQLite):

1. employees
   - id            INTEGER PRIMARY KEY
   - name          TEXT
   - department_id INTEGER (FK → departments.id)
   - role          TEXT        (e.g. 'Software Engineer', 'HR Generalist')
   - level         TEXT        ('Junior', 'Mid', 'Senior', 'Lead', 'Manager')
   - salary        REAL        (annual, EUR)
   - hire_date     TEXT        (ISO format: YYYY-MM-DD)
   - manager_id    INTEGER     (FK → employees.id, nullable)
   - location      TEXT        ('Madrid', 'Barcelona', 'Remote', 'London')
   - status        TEXT        ('active', 'terminated')

2. departments
   - id            INTEGER PRIMARY KEY
   - name          TEXT        (e.g. 'Engineering', 'HR', 'Sales', 'Finance')
   - budget        REAL        (annual department budget, EUR)
   - head_id       INTEGER     (FK → employees.id — department head)

3. performance_reviews
   - id            INTEGER PRIMARY KEY
   - employee_id   INTEGER     (FK → employees.id)
   - review_year   INTEGER
   - score         REAL        (1.0 – 5.0)
   - rating_label  TEXT        ('Needs Improvement', 'Meets Expectations',
                                'Exceeds Expectations', 'Outstanding')
   - reviewer_id   INTEGER     (FK → employees.id)

4. job_postings
   - id            INTEGER PRIMARY KEY
   - title         TEXT
   - department_id INTEGER     (FK → departments.id)
   - posted_date   TEXT        (ISO format)
   - status        TEXT        ('open', 'filled', 'cancelled')
   - applicants    INTEGER

USEFUL QUERY PATTERNS:
- Headcount by department: SELECT d.name, COUNT(*) FROM employees e JOIN departments d ON e.department_id=d.id WHERE e.status='active' GROUP BY d.name
- Average salary by level:  SELECT level, ROUND(AVG(salary),2) FROM employees WHERE status='active' GROUP BY level ORDER BY AVG(salary) DESC
- Attrition rate:           SELECT ROUND(100.0*SUM(CASE WHEN status='terminated' THEN 1 ELSE 0 END)/COUNT(*),2) AS attrition_pct FROM employees
- Top performers 2024:      SELECT e.name, p.score FROM performance_reviews p JOIN employees e ON p.employee_id=e.id WHERE p.review_year=2024 ORDER BY p.score DESC LIMIT 10
"""

    def _get_connection(self):
        """Return a READ-ONLY SQLite connection using the configured HR_DB_URI.

        Opening in mode=ro is defense-in-depth: even if the LLM-generated SQL
        slips past _is_select_only(), SQLite itself will refuse any write.
        """
        try:
            db_uri = current_app.config.get("HR_DB_URI", "")
            # Strip SQLAlchemy prefix if present
            db_path = db_uri.replace("sqlite:///", "")
            if db_path == ":memory:":
                conn = sqlite3.connect(db_path)
            else:
                conn = sqlite3.connect(f"file:{quote(db_path)}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            print(f"❌ Error connecting to HR DB: {e}")
            return None

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        conn = None
        try:
            # ── 1. Let LLM generate the SQL ──────────────────────────────────
            user_sql_context = self.project_settings.get("sql_context", "") or self.DB_SCHEMA_CONTEXT

            llm = ChatOpenAI(model_name=self.model_name, temperature=0)

            system_prompt = f"""You are an expert SQL analyst for HR data.
Given the database schema below, write a single valid SQLite SELECT query to answer the user's question.
Return ONLY the SQL — no explanation, no markdown fences.

{user_sql_context}

Rules:
- Use only the tables and columns listed above.
- Always filter employees by status='active' unless the question explicitly asks about terminated employees.
- Round float results to 2 decimal places.
- Use meaningful column aliases (e.g. AS avg_salary).
- NEVER use DROP, DELETE, UPDATE, INSERT or any DDL/DML."""

            # ── 2. Execute the SQL, with bounded self-correction ──────────────
            conn = self._get_connection()
            if not conn:
                return "❌ Could not connect to the HR database."
            cursor = conn.cursor()

            messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
            generated_sql, rows, columns, last_error = "", None, None, None

            for attempt in range(MAX_SQL_ATTEMPTS):
                generated_sql = llm.invoke(messages).content.strip().strip("```sql").strip("```").strip()
                print(f"🔍 Generated SQL (attempt {attempt + 1}/{MAX_SQL_ATTEMPTS}):\n{generated_sql}")

                # ── Safety check: only single read-only SELECT/CTE statements ─
                if not _is_select_only(generated_sql):
                    return "❌ Generated query rejected: only single read-only SELECT statements are allowed."

                try:
                    cursor.execute(generated_sql)
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    last_error = None
                    break
                except sqlite3.Error as e:
                    last_error = str(e)
                    if attempt == MAX_SQL_ATTEMPTS - 1:
                        break
                    # ✅ Self-correction: pasamos el error real de SQLite al LLM para que corrija la query
                    messages.append(AIMessage(content=generated_sql))
                    messages.append(HumanMessage(
                        content=(
                            f"That query failed with this SQLite error:\n{last_error}\n\n"
                            "Fix it and return ONLY the corrected SQL."
                        )
                    ))

            if last_error is not None:
                print(f"❌ SQL query failed after {MAX_SQL_ATTEMPTS} attempts: {last_error}")
                return "I couldn't run a valid query for that question. Try rephrasing it or being more specific."

            if not rows:
                return "No results found for your query."

            # ── 3. Format results as a readable table ─────────────────────────
            header = " | ".join(columns)
            separator = "-" * len(header)
            data_rows = "\n".join(" | ".join(str(r[c]) for c in columns) for r in rows)

            table = f"{header}\n{separator}\n{data_rows}"

            # ── 4. Ask LLM for a human-readable interpretation ────────────────
            interpret_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an HR analyst. Interpret the SQL query result below "
                    "in 2-4 clear sentences. Mention key numbers, trends, or notable findings. "
                    "Do not repeat the table verbatim."
                )),
                ("user", f"Original question: {query}\n\nQuery result:\n{table}"),
            ])

            interpretation = (interpret_prompt | llm).invoke({}).content

            return f"**Query result:**\n```\n{table}\n```\n\n**Interpretation:** {interpretation}"

        except Exception as e:
            print(f"❌ HRDatabaseTool error: {e}")
            return f"Error executing HR database query: {str(e)}"
        finally:
            if conn:
                conn.close()


# Alias para compatibilidad con routes.py
SQLDatabaseTool = HRDatabaseTool