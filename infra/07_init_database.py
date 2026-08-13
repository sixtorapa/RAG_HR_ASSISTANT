"""
Step 7 — create the schema on RDS and connect the Lambda to it.

Run FROM OUTSIDE, not from the Lambda, and that is deliberate: the function then
never needs permission to alter the schema. It only reads and writes rows. That
is what makes running startup.sh inside Lambda unnecessary.

⚠️ `flask db upgrade` cannot be run against an EMPTY database. This project's
migrations are ALTER TABLE statements that assume the tables already exist, so
the first one fails with:

    relation "chat_session" does not exist

The schema has to be created from scratch with db.create_all() and the
migrations then MARKED as applied (stamp head). It is exactly the branch
startup.sh already had.

Usage:  python infra/07_init_database.py
"""

import os
import sys

from _common import (
    FUNCTION_NAME,
    REPO_ROOT,
    client,
    require_credentials,
)

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

if not os.environ.get("DATABASE_URL"):
    sys.exit("✗ No DATABASE_URL in .env. Has the RDS finished being created (step 6)?")

print(f"  destino: {os.environ['DATABASE_URL'].split('@')[-1]}")

from flask_migrate import stamp, upgrade  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import create_app, db  # noqa: E402
from config import Config  # noqa: E402

app = create_app(Config)

with app.app_context():
    tables = set(db.inspect(db.engine).get_table_names())
    print(f"  tables at start: {sorted(tables) or '(none)'}")

    if "user" in tables:
        print("  existing database → applying pending migrations")
        upgrade()
    else:
        print("  empty database → creating the schema from scratch")
        db.session.execute(text("DROP TABLE IF EXISTS alembic_version"))
        db.session.commit()
        db.create_all()
        stamp()   # mark the migrations as already applied
        print("  schema created and migrations stamped (stamp head)")

    print(f"  tables now: {', '.join(sorted(db.inspect(db.engine).get_table_names()))}")

    from app.models import User
    if not User.query.filter_by(username="admin").first():
        u = User(username="admin", role="admin", is_active=True)
        u.set_password("admin1234")
        db.session.add(u)
        db.session.commit()
        print("  usuario admin creado")
    else:
        print("  admin user already existed")

# ── Connect the Lambda to the database ───────────────────────────────────────
# The secret-bearing variables are added here; step 4 deliberately left them out.
require_credentials()
lam = client("lambda")
env = lam.get_function_configuration(FunctionName=FUNCTION_NAME)["Environment"]["Variables"]
env.update({
    "DATABASE_URL": os.environ["DATABASE_URL"],
    # Embeddings stay on OpenAI by deliberate decision: changing them would
    # invalidate the vector index and the evaluation numbers already measured.
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "SECRET_KEY": os.environ.get("SECRET_KEY", "cambiar-en-produccion"),
    # Cost cap: Bedrock is pay-per-use with NO automatic ceiling, and AWS budget
    # alarms warn but do not stop. The counter lives in Postgres, not in memory:
    # each container has its own, and N parallel containers would multiply
    # a RAM counter.
    "DAILY_QUESTION_LIMIT": os.environ.get("DAILY_QUESTION_LIMIT", "15"),
})
lam.update_function_configuration(
    FunctionName=FUNCTION_NAME, Environment={"Variables": env}
)
lam.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
print("\n  ✓ Lambda connected to the database")

# The vector store path is configuration (UP_VECTOR_STORE_PATH), not a database
# column, so pointing the function at /tmp only takes an environment variable.
