# ─────────────────────────────────────────────────────────────────────────────
# HR Knowledge Base Assistant — Dockerfile
# Target: Railway / any Linux container host
# Python 3.11 slim · CPU-only torch · No Tesseract
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libmagic1   → python-magic (file type detection)
# libgomp1    → OpenMP required by onnxruntime / FlashRank
# curl        → healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
# Step 1: torch CPU-only FIRST (avoids pulling the full CUDA build, saves ~1.2 GB)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: rest of the requirements
# --extra-index-url ensures torch stays CPU-only even if sentence-transformers
# or any other dep tries to resolve torch again (avoids pulling nvidia_* CUDA wheels)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Runtime environment ───────────────────────────────────────────────────────
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Port Railway will use (Railway sets $PORT automatically)
ENV PORT=8080
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
# gunicorn is already in your requirements
# --workers 2 · --threads 4 → safe for a 512MB Railway free container
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "run:app"]