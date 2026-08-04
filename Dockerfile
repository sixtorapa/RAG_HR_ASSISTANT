# ─────────────────────────────────────────────────────────────────────────────
# HR Knowledge Base Assistant — Dockerfile
# Target: Railway, or any Linux container host.
# Python 3.12 slim, no torch. The Lambda image is separate: see Dockerfile.lambda.
# ─────────────────────────────────────────────────────────────────────────────

# 3.12 rather than 3.11: langchain-aws pins numpy<2 only below 3.12, which
# clashes with the numpy>=2.3.0 in requirements-prod.txt. On 3.11 the image
# resolved a different dependency graph from the one the tests ran against —
# "works on my machine" in its purest form.
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# torch is NOT installed. Its only use was FlashrankRerank in qa_chain.py, and
# the reranker is off by default (FLASHRANK_ENABLED=0) because measuring it
# dropped context precision from 0.86 to 0.64. It dragged in hundreds of MB, and
# on Lambda image size is cold-start time.
# To bring FlashRank back, reinstall it from the CPU index:
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# ── Application source ───────────────────────────────────────────────────
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