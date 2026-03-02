#!/bin/bash
set -e

echo ">>> Initializing database..."
python - <<'PYEOF'
from app import create_app, db
app = create_app()
with app.app_context():
    db.create_all()
    print("✅ All tables created.")
PYEOF

echo ">>> Stamping migrations as current..."
flask db stamp head

echo ">>> Seeding HR database..."
python seed_hr_db.py

echo ">>> Creating admin user if not exists..."
python create_admin.py

echo ">>> Starting gunicorn..."
echo ">>> ENV: WEB_CONCURRENCY=${WEB_CONCURRENCY:-<unset>} GUNICORN_CMD_ARGS=${GUNICORN_CMD_ARGS:-<unset>} GUNICORN_WORKERS=${GUNICORN_WORKERS:-<unset>} GUNICORN_THREADS=${GUNICORN_THREADS:-<unset>}"

# Railway a veces mete WEB_CONCURRENCY; nosotros CAPAMOS por defecto a 1
# (si quieres más, que sea explícito con GUNICORN_WORKERS)
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
