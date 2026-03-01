#!/bin/bash
set -e

echo ">>> Running database migrations..."
flask db upgrade

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
