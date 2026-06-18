#!/bin/bash
set -e

echo ">>> Checking database state..."
# Las migraciones antiguas de este proyecto son ALTER TABLE que asumen que la
# tabla ya existe (se crearon pensando en una BD ya inicializada por
# db.create_all(), no desde cero). Por eso no podemos simplemente correr
# "flask db upgrade" siempre: en una base de datos nueva, replicaría esas
# migraciones contra tablas que create_all() ya habría creado con el esquema
# final, fallando con "column already exists". Así que decidimos según si la
# tabla "user" ya existe:
#   - No existe (BD nueva)      -> create_all() + stamp head (crea todo de una vez)
#   - Ya existe (BD persistente) -> flask db upgrade (aplica solo lo pendiente de verdad)
python - <<'PYEOF'
import sys
from app import create_app, db
from sqlalchemy import inspect
app = create_app()
with app.app_context():
    sys.exit(0 if inspect(db.engine).has_table("user") else 1)
PYEOF
DB_ALREADY_INITIALIZED=$?

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
