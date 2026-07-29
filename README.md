# HR Knowledge Base Assistant

> A RAG system for HR teams — hybrid retrieval, a deterministic agent router, text-to-SQL
> with guardrails, and a measured evaluation suite. Runs on two clouds from one codebase.

![Docker](https://img.shields.io/badge/docker-ready-blue)
![AWS](https://img.shields.io/badge/AWS-Lambda%20%2B%20Bedrock-orange)
![LangSmith](https://img.shields.io/badge/observability-LangSmith-purple)

**Live demo (AWS):** https://81a5fl8aji.execute-api.eu-west-1.amazonaws.com
Sign in with `admin` / `admin1234`. It is a public demo on synthetic HR data — there is
nothing confidential behind that login, by design.

---

## What it does

Employees ask questions in natural language, and the system decides where the answer
lives before answering:

| Source | Example |
|---|---|
| **Documents** (PDF, DOCX, PPTX, XLSX) | *"What is our remote-work policy?"* |
| **Structured HR data** (SQL) | *"What is the average salary in Engineering?"* |

Answers come back with source citations, and every user only ever sees the departments
they are allowed to see.

---

## Architecture

```
User query
    │
    ▼
_settings_with_acl()          ← RBAC injected BEFORE retrieval, never after
    │
    ▼
AgentRouter — three paths
    │  1. deterministic fast-path (greetings, smalltalk) → no LLM call at all
    │  2. clear SQL/Excel intent   → tool forced, no LLM call
    │  3. ambiguous                → LLM decides via bind_tools
    │
    ├─► chat_with_documents ─► native metadata prefilter
    │                       ─► two-pass document shortlist
    │                       ─► hybrid retrieval: BM25 + vector (RRF, 0.55/0.45)
    │                       ─► ConversationalRetrievalChain → answer + sources
    │
    ├─► query_hr_database   ─► LLM-generated SQL, SELECT-only, read-only connection
    │
    └─► summarise_document  ─► full-document summarisation
```

---

## Two deployments, one codebase

The provider and the host are **independent choices**. Either can change without the other.

| | Railway | AWS |
|---|---|---|
| Compute | Container, always on | Lambda container behind API Gateway |
| Generation | OpenAI `gpt-4o` / `gpt-4o-mini` | **Bedrock** — Claude Sonnet 4.6 / Haiku 4.5 |
| Embeddings | OpenAI `text-embedding-3-small` | OpenAI (unchanged — see below) |
| App database | PostgreSQL | RDS PostgreSQL |
| Entry point | gunicorn (`startup.sh`) | `lambda_handler.py` via `apig-wsgi` |

Switching provider is one environment variable:

```bash
LLM_PROVIDER=openai    # default
LLM_PROVIDER=bedrock
```

Before this existed, `ChatOpenAI` and `OpenAIEmbeddings` were instantiated inline in
**19 places across 10 modules**. They now all go through `app/rag_logic/llm_factory.py`,
which is the only file in the repo that names a provider.

**Embeddings deliberately stay on OpenAI.** The vectors in Chroma were built with
`text-embedding-3-small`; changing the embedding model invalidates the whole index and
forces a full re-ingest, and would also invalidate the evaluation numbers below. The
migration is incremental, not a big bang.

---

## What running this on Lambda actually taught me

Three failures that only appear once it is deployed. All have the same root: in Lambda
`/var/task` is read-only and only `/tmp` can be written.

1. **A Chroma vector store cannot be served read-only from the image.** The common advice
   is to bake it in and read it in place. Chroma opens its SQLite read-write even for
   queries because it needs its journal, so retrieval fails with
   `attempt to write a readonly database (code: 8)` while the index sits right there.
   It is copied to `/tmp` during the init phase — 8.7 MB, once per container.
2. **Flask's `instance_path` lives next to the code.** The conversational memory store
   writes there, so `/ask` returned 500 until it was pointed at `/tmp`.
3. **Lambda rejects OCI image manifests.** It needs Docker v2 schema 2, and
   `--provenance=false` alone is not enough — the push must set `oci-mediatypes=false`.

Measured on the deployed function:

| | |
|---|---|
| Cold start (image already cached on the host) | 4.7 s init |
| Warm request | 0.06 s |
| Memory used / allocated | 308 MB / 3008 MB |

Memory is over-allocated on purpose: in Lambda, CPU scales with memory, and lowering it
lengthens the cold start.

**Known limit:** a `/ask` round trip takes ~27 s, against API Gateway's 29 s ceiling. The
first query on a cold container exceeds it and returns 503. The fixes are known and not
yet applied — skipping the reasoning agent when there is a single result, and truncating
after RRF fusion, which currently lets ~50 chunks through instead of 28.

---

## Design decisions worth defending

**RBAC, fail-closed.** A user's allowed departments are ANDed with any functional filter,
in both retrieval passes. An empty department list becomes
`{"department": "__no_access__"}` — zero results, not full access. Default deny is the
whole point: treating "empty" as "unrestricted" passes tests and leaks in production.
The ACL is also part of the chain cache key, so two users with different permissions can
never share a cached retriever.

**A router where two of three paths never reach the LLM.** A "hello" should not cost a
model call. Less cost, less latency, and the same input always takes the same route.

**Hybrid retrieval, measured rather than assumed.** `EnsembleRetriever` fuses BM25 and
vector search by **rank** (weighted Reciprocal Rank Fusion, `w/(60+rank)`), not by score —
cosine and BM25 are not comparable in magnitude, but rank is.

**The reranker is off, and that is the interesting part.** FlashRank was added, measured,
and dropped: it took context precision from 0.86 to 0.64 and caused OOM on the production
tier. It stays opt-in behind `FLASHRANK_ENABLED`, off by default, with the numbers
committed so nobody "fixes" it back on.

**Text-to-SQL that cannot write.** Single `SELECT`/`WITH` statements only, read-only
connection, sensitive columns blocked, with a self-correction retry loop.

**PII detection without ML.** IBAN, card and DNI/NIE are validated by checksum, before the
input reaches the model or the database. A checksum has no random false positives and
never needs retraining.

---

## Evaluation

`evaluation/evaluate_rag.py` — a golden dataset of 25 questions (21 RAG, 4 SQL) over the
real documents, three pipeline configurations, four RAGAS metrics, results committed as
JSON and CSV.

| Config | Context precision | Context recall | Faithfulness | Answer relevancy |
|---|---|---|---|---|
| `vector_only` | 0.858 | 0.881 | 0.857 | 0.763 |
| **`bm25_vector`** ← shipped | **0.861** | **0.937** | 0.857 | **0.812** |
| `full_pipeline` (FlashRank) | 0.638 | 0.818 | 0.865 | 0.767 |

Adding BM25 buys **+5.6 points of recall** and **+4.9 of answer relevancy** at no cost to
precision. It does not retrieve better things; it stops losing the ones semantic search
missed.

25 questions is directional signal for comparing configurations, not a fine estimate.
The harness is the expensive part and it is already built.

---

## Quick start

```bash
git clone https://github.com/sixtorapa/RAG_HR_ASSISTANT.git
cd RAG_HR_ASSISTANT

python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-prod.txt

printf 'OPENAI_API_KEY=your-key\n' > .env

python seed_hr_db.py     # toy HR SQLite database
pytest                   # 61 tests; OpenAI is mocked, no key needed
python run.py            # http://localhost:5001
```

Or with Docker:

```bash
docker compose up --build     # http://localhost:8080
```

**Python 3.12 is required.** `langchain-aws` pins `numpy<2` below 3.12, which conflicts
with this project's `numpy>=2.3.0`.

### Running against Bedrock

```env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-west-1
```

Model IDs are **inference profiles** (`eu.anthropic.claude-sonnet-4-6`), not bare model
IDs. The `eu.` prefix keeps inference inside the EU, which matters for a system holding
employee data. A bare `anthropic.claude-…` returns `ValidationException`.

---

## Observability

`observability.py` enables LangSmith tracing from environment variables and is called at
startup:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=hr-kb-assistant
```

`cost_calculator.py` computes per-query cost from token usage. Prompt tokens dominate in
RAG — the retrieved chunks are large and the answer is short — which makes `k_base` a
cost decision, not only a quality one.

---

## Project structure

```
├── run.py                      # Flask entry point (Railway)
├── lambda_handler.py           # Lambda entry point (apig-wsgi)
├── config.py                   # All configuration via env vars
├── Dockerfile                  # Railway image
├── Dockerfile.lambda           # Lambda image (AWS base, vector store baked in)
│
├── app/
│   ├── models.py               # SQLAlchemy: User, Project, ChatSession, Message
│   ├── main/routes.py          # Endpoints; /ask orchestrates router → tools → chain
│   └── rag_logic/
│       ├── llm_factory.py      # ← the only file that names a provider
│       ├── agent_router.py     # Three-path routing
│       ├── qa_chain.py         # RBAC, prefilter, two-pass, hybrid retrieval, cache
│       ├── ingester.py         # Ingestion: loaders, chunking, enrichment, indexing
│       ├── custom_loaders.py   # PDF/PPTX loaders with table extraction and OCR
│       ├── bm25_index.py       # BM25 built at ingest, persisted, loaded at query
│       ├── sql_tool.py         # Text-to-SQL with guardrails
│       ├── pii_guard.py        # Checksum-based PII detection
│       └── cost_calculator.py  # Per-query cost from token usage
│
├── evaluation/evaluate_rag.py  # RAGAS suite, golden dataset, 3 configurations
├── infra/                      # how the AWS side was built, in order
└── tests/                      # 61 tests, OpenAI mocked
```

### infra/

The seven scripts that created the AWS deployment, numbered in the order they were
run: the IAM execution role, the ECR repository and image push, the Lambda function,
the API Gateway endpoint, the RDS instance, and the schema initialisation.

They are honest about what they are: **imperative scripts, not infrastructure as
code**. There is no state file, no plan, and nothing detects drift. Turning them into
Terraform is mechanical — every resource and parameter is already written down — and
has not been done. `infra/README.md` says so explicitly, along with the security
trade-off behind keeping the Lambda outside the VPC and the teardown commands.

---

## Honest limitations

- **CI, not CD.** `.github/workflows/ci.yml` runs ruff, pytest and a Docker build with
  `push: false`. There is no deploy job; Railway deploys from its own git integration.
  Its `docker-build` job also only `needs: lint`, so it can build with tests red.
- The Lambda deployment has the ~27 s / 29 s ceiling described above.
- Chunking is **recursive by tokens**, not semantic. The semantic part is an LLM
  enrichment pass at ingest time that prepends a headline and summary before embedding.
- Parent/child chunk metadata exists (`parent_chunk_id`) but child→parent expansion is
  not wired up yet.
- BM25 tokenises on whitespace only — no stemming, no accent normalisation. In Spanish
  that costs more than in English.
- The router's fast-path matches substrings without word boundaries, so `"suma"` matches
  inside `"consumar"`. Since the forced path skips the LLM, a false positive routes
  without a safety net.
- `evaluate_rag.py` reimplements retrieval instead of calling the production chain, so
  its numbers do not measure the shipped pipeline exactly.

---

## License

No license file yet — all rights reserved for now.
