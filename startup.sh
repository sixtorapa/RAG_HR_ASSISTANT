#!/bin/bash
set -e

echo ">>> Initializing database..."
python - <<'EOF'
from app import create_app, db
app = create_app()
with app.app_context():
    db.create_all()
    print("✅ All tables created.")
EOF

echo ">>> Stamping migrations as current..."
flask db stamp head

echo ">>> Seeding HR database..."
python seed_hr_db.py

echo ">>> Creating admin user if not exists..."
python create_admin.py

echo ">>> Starting gunicorn..."
exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 2 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    "run:app"
