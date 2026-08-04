#!/bin/bash
set -e

echo ">>> Checking database state..."
# The older migrations in this project are ALTER TABLE statements that assume the
# table already exists: they were written against a database initialised by
# db.create_all(), not one built from scratch. So "flask db upgrade" cannot be
# run unconditionally — on a fresh database it would replay those migrations
# against tables create_all() had already created with the final schema, failing
# with "column already exists". The branch is decided by whether "user" exists:
#   - absent  (fresh database)      -> create_all() + stamp head, all at once
#   - present (existing database)   -> flask db upgrade, applying what is pending
# set +e/-e: this check uses the exit code (0/1) as a signal on purpose. With
# "set -e" active, an exit 1 here would kill the script before "$?" can be read,
# even though it is not an error.
set +e
python - <<'PYEOF'
import sys
from app import create_app, db
from sqlalchemy import inspect
app = create_app()
with app.app_context():
    sys.exit(0 if inspect(db.engine).has_table("user") else 1)
PYEOF
DB_ALREADY_INITIALIZED=$?
set -e

if [ "$DB_ALREADY_INITIALIZED" -eq 0 ]; then
    echo ">>> Existing database — applying pending migrations..."
    flask db upgrade
else
    echo ">>> Fresh database — creating full schema..."
    python - <<'PYEOF'
from app import create_app, db
app = create_app()
with app.app_context():
    db.create_all()
    print("✅ All tables created.")
PYEOF
    flask db stamp head
fi

echo ">>> Seeding HR database..."
python seed_hr_db.py

echo ">>> Creating admin user if not exists..."
python create_admin.py

echo ">>> Starting gunicorn..."
echo ">>> ENV: WEB_CONCURRENCY=${WEB_CONCURRENCY:-<unset>} GUNICORN_CMD_ARGS=${GUNICORN_CMD_ARGS:-<unset>} GUNICORN_WORKERS=${GUNICORN_WORKERS:-<unset>} GUNICORN_THREADS=${GUNICORN_THREADS:-<unset>}"

# Railway sometimes injects WEB_CONCURRENCY. Default to a single worker here;
# more has to be asked for explicitly through GUNICORN_WORKERS.
WORKERS="${GUNICORN_WORKERS:-1}"
THREADS="${GUNICORN_THREADS:-1}"

exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers "${WORKERS}" \
    --threads "${THREADS}" \
    --timeout 90 \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile - \
    "run:app"
