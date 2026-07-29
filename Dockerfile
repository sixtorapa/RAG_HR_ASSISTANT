# ─────────────────────────────────────────────────────────────────────────────
# HR Knowledge Base Assistant — Dockerfile
# Target: Railway / any Linux container host
# Python 3.12 slim · sin torch · sin Tesseract
# (la imagen de Lambda es otra: ver Dockerfile.lambda)
# ─────────────────────────────────────────────────────────────────────────────

# 3.12 y no 3.11: langchain-aws exige numpy<2 solo en Python <3.12, lo que
# choca con el numpy>=2.3.0 de requirements-prod.txt. El .venv donde pasan los
# 61 tests ya es 3.12.13 — con 3.11 la imagen resolvía un grafo de dependencias
# distinto del que se probaba en local.
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
# torch NO se instala: su único uso era FlashrankRerank en qa_chain.py, y el
# reranker está desactivado por defecto (FLASHRANK_ENABLED=0) porque al medirlo
# hundía la context precision de 0.86 a 0.64. Arrastraba cientos de MB, y en
# Lambda el tamaño de la imagen es tiempo de arranque en frío.
# Para reactivar FlashRank hay que volver a añadirlo con el índice CPU:
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Runtime environment ───────────────────────────────────────────────────────
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080


EXPOSE 8080

# ── Startup script ───────────────────────────────────────────────────────────
COPY startup.sh .
RUN chmod +x startup.sh

CMD ["./startup.sh"]