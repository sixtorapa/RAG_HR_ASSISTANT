# HR Knowledge Base Assistant

> A production-grade RAG system for HR teams — built with LangChain, Flask, and ChromaDB.

![CI](https://github.com/<YOUR_USERNAME>/hr-kb-assistant/actions/workflows/ci.yml/badge.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![LangSmith](https://img.shields.io/badge/observability-LangSmith-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Employees and HR managers can ask natural-language questions about company knowledge in two ways:

| Source | Examples |
|--------|---------|
| **Documents** (PDFs, DOCX, PPTX) | "What is our remote-work policy?" / "Summarise the onboarding handbook" |
| **Structured HR data** (SQLite / Postgres) | "What is the average salary in Engineering?" / "Show attrition by department" |

The system routes each question to the right tool automatically via an **Agent Router**, then retrieves, reranks, and synthesises an answer with source citations.

---

## Architecture

```
User Query
    │
    ▼
AgentRouter (OpenAI function-calling)
    │
    ├─► chat_with_documents ──► Hybrid Retriever (BM25 + Vector)
    │                               │
    │                           EnsembleRetriever
    │                               │
    │                           Flashrank Reranker
    │                               │
    │                           ConversationalQAChain ──► Answer + Sources
    │
    ├─► query_hr_database ──► LLM-generated SQL ──► SQLite/Postgres ──► Interpreted result
    │
    └─► summarise_document ──► Full-doc RAG summarisation
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o / GPT-4o-mini (OpenAI) |
| Orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | OpenAI `text-embedding-3-small` |
| Keyword search | BM25 (rank_bm25) |
| Reranking | Flashrank |
| Web framework | Flask |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Observability | LangSmith |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions → Railway |

---

## Quick start

### 1. Clone & configure

```bash
git clone https://github.com/<YOUR_USERNAME>/hr-kb-assistant.git
cd hr-kb-assistant
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

App will be available at `http://localhost:5001`.

### 3. Run locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python seed_hr_db.py        # creates toy HR SQLite database
flask run --port 5001
```

---

## Enabling LangSmith tracing

1. Create a free account at [smith.langchain.com](https://smith.langchain.com)
2. Add to your `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=hr-kb-assistant
```

Every LLM call, tool invocation and retrieval step will appear in your LangSmith dashboard with latency and token cost.

---

## Project structure

```
hr-kb-assistant/
├── run.py                  # Flask entry point
├── config.py               # All config via env vars
├── seed_hr_db.py           # Creates toy SQLite HR database
├── observability.py        # LangSmith init helper
│
├── app/
│   ├── models.py           # SQLAlchemy models
│   ├── routes.py           # Flask blueprints
│   └── rag_logic/
│       ├── agent_router.py     # Orchestrator — routes to tools
│       ├── qa_chain.py         # Hybrid retrieval + reranking
│       ├── sql_tool.py         # Text-to-SQL over HR database
│       ├── prompt_templates.py # Modular system prompts
│       ├── ingester.py         # Document ingestion pipeline
│       └── tools.py            # LangChain tool wrappers
│
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml          # Lint → Test → Docker build → Deploy
```

---

## Roadmap

- [ ] RAGAS evaluation suite for retrieval quality metrics
- [ ] OCR support for scanned PDFs (Tesseract)
- [ ] Role-based access control per document folder
- [ ] Streaming responses via SSE
