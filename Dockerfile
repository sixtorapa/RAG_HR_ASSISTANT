# ─────────────────────────────────────────────────────────────────────────────
# HR Knowledge Base Assistant — Dockerfile
# Target: Railway / any Linux container host
# Python 3.11 slim · CPU-only torch · No Tesseract
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Runtime environment ───────────────────────────────────────────────────────
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# ── FlashRank: modelo ligero + cache fijo para evitar descarga en runtime ─────
# MiniLM-L-12-v2 es mucho más ligero que MultiBERT (~40MB vs ~100MB)
ENV FLASHRANK_CACHE_DIR=/opt/flashrank \
    FLASHRANK_MODEL_NAME=ms-marco-MiniLM-L-12-v2

RUN mkdir -p /opt/flashrank && \
    python -c "import os; from flashrank import Ranker; Ranker(model_name=os.environ['FLASHRANK_MODEL_NAME'], cache_dir=os.environ['FLASHRANK_CACHE_DIR'])"

EXPOSE 8080

# ── Startup script ───────────────────────────────────────────────────────────
COPY startup.sh .
RUN chmod +x startup.sh

CMD ["./startup.sh"]
