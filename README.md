# HR Knowledge Base Assistant

> A RAG system for HR teams: hybrid retrieval, a deterministic agent router, text-to-SQL
> with guardrails, and every design decision backed by a measurement. Runs on two clouds
> from one codebase.

![Docker](https://img.shields.io/badge/docker-ready-blue)
![AWS](https://img.shields.io/badge/AWS-Lambda%20%2B%20Bedrock-orange)
![LangSmith](https://img.shields.io/badge/observability-LangSmith-purple)
![Tests](https://img.shields.io/badge/tests-295%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

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
POST /ask
   │
   ├── daily quota  ──────────► 429    both guardrails run before the first LLM call
   ├── PII detection ─────────► 400    and before anything is written to the database
   │
   ├── RBAC injected into the retriever settings — before retrieval, never after
   │
   ▼
AgentRouter — three paths
   │  1. deterministic fast-path (greetings, smalltalk)  → answers directly, no LLM
   │  2. clear SQL / Excel intent                        → tool forced, no LLM
   │  3. anything ambiguous                              → LLM decides via bind_tools
   ▼
one dispatch loop over the chosen tools
   │
   ├─► chat_with_documents ─► retrieval (below) ─► answer + sources
   ├─► query_hr_database   ─► LLM-generated SQL, SELECT-only, read-only connection
   ├─► analista_de_excel   ─► pandas agent, bounded iterations and timeout
   └─► summarise_document  ─► full-document summarisation
   ▼
ReasoningAgent formats the result  →  persisted  →  JSON with sources
```

The router decides and names; the dispatch loop executes. An explicit user override
(`SQL: …`, `AMBAS - …`, asking for a summary) produces the same list of tool calls the
router would have returned, so there is one execution path rather than one per mode.

---

## Retrieval

### Two chunk sizes, each used for what it is good at

Ingestion produces two granularities of the same document: **macro** chunks built from
whole pages (~350 words) and **micro** chunks of 250 tokens that carry a
`parent_chunk_id`.

Search runs against the micro chunks, whose embeddings are specific and undiluted. What
reaches the model is the **parent**, which carries the full page. Several matching
children of one document collapse into a single entry, and each parent takes the rank of
its first matching child.

```
RETRIEVAL_CHUNK_TYPE=micro   (default; also accepts macro or all)
PARENT_EXPANSION=1           (default)
```

Both are part of the chain cache key, so changing either does not serve a stale chain.

### Hybrid search, fused by rank

`EnsembleRetriever` combines BM25 and vector search with weights 0.55 / 0.45. It fuses by
**rank**, not by score — weighted Reciprocal Rank Fusion, `w/(60+rank)`. Cosine similarity
is bounded and BM25 is not, so their magnitudes are not comparable; their ranks are. The
vector leg uses MMR (`lambda_mult=0.55`) for diversity, and the BM25 index is built during
ingestion and persisted, not rebuilt per query.

### Two-pass document shortlist

The first pass searches broadly and counts chunk votes per document; the second re-searches
scoped to the winners using a native metadata prefilter. On a generic question — *"how many
days of sick leave do I get?"* — flat retrieval blends fragments of five documents and the
model synthesises an answer that exists in none of them.

### Access control, fail-closed

A user's allowed departments are ANDed with any functional filter, in both retrieval
passes. An empty department list becomes `{"department": "__no_access__"}` — zero results,
not full access. Default deny is the point: treating "empty" as "unrestricted" passes tests
and leaks in production. The ACL is part of the chain cache key, so two users with
different permissions can never share a cached retriever.

---

## Guardrails

**PII detection without ML.** IBAN, card and DNI/NIE are validated by checksum — mod-97,
Luhn, and the Spanish control letter — before the input reaches the model or the database.
A checksum has no random false positives and never needs retraining. The policy is to
block, not to redact and continue: if a redactor misses one field the data still escapes.

**Text-to-SQL that cannot write.** Single `SELECT`/`WITH` statements only, on a connection
opened `mode=ro` so SQLite itself refuses a write even if the syntax check is fooled.
Columns tied to departments the user cannot access are blocked deterministically, not by
asking the model nicely in the prompt. Failed queries are retried up to three times with
the real SQLite error fed back.

**A daily question quota.** Bedrock is pay-per-use with no automatic ceiling and AWS budget
alarms warn rather than stop, so the public demo caps questions per day. The counter lives
in the database, not in process memory: in Lambda each container has its own memory, so a
RAM counter would apply the cap per container and N containers would multiply it.

---

## Evaluation

`evaluation/evaluate_pipeline.py` runs the **deployed chain** — two-pass, ACL, hybrid
retrieval, parent expansion — over a golden dataset of 21 RAG questions written against
the real documents, and scores it with RAGAS.

Figures are the mean of three runs, with the spread across them, because a single run
does not settle these numbers.

| Metric | Score | Spread | What it measures |
|---|---|---|---|
| Context precision | **0.818** | ±0.005 | Of what was retrieved, how much is relevant |
| Context recall | 0.762 | ±0.000 | Whether the context holds everything the answer needs |
| **Faithfulness** | **0.946** | ±0.011 | Whether the answer is anchored in the context |
| Answer relevancy | 0.859 | ±0.011 | Whether it answers what was asked |

Context recall lands on exactly 0.762 in every run: the metric is quantised per question,
so that is 16 of 21, and the gap is one specific question rather than noise.

A query costs about **$0.0003** and 2,300 prompt tokens. Prompt tokens dominate in RAG —
the retrieved chunks are large and the answer is short — which makes retrieval breadth a
cost decision, not only a quality one.

Twenty-one questions is directional signal for comparing configurations, not a fine
estimate. Recall is coarse at that size: one question is worth 4.8 points.

`evaluation/evaluate_rag.py` is a separate harness that compares three retrieval
configurations on its own retriever. It is what established that FlashRank reranking took
context precision from 0.86 to 0.64 on that setup, which is why the reranker stays behind
`FLASHRANK_ENABLED`, off, with the numbers committed.

---

## Testing

```bash
pytest        # 295 tests, OpenAI mocked — no API key, no cost
```

The tests assert properties rather than implementation, and the ones worth reading are the
ones covering the parts the design leans on:

| | Asserts |
|---|---|
| `test_guardrails.py` | An empty department list must deny **and** must not produce a null filter — returning `None` there would search the whole corpus. An IBAN with a broken check digit must **not** be detected, which is the entire argument for checksums over pattern matching. A detected value never appears whole in a finding, since findings go to the log. |
| `test_router.py` | Each case really is an instance of substring containment *before* asserting that word matching rejects it, so the test cannot quietly become decorative. The first two paths must never invoke the LLM. |
| `test_ingester.py` | No page with text disappears when chunking. Every micro chunk resolves to a parent that exists. Chunk ids are stable across runs, or incremental ingestion silently stops being incremental. |
| `test_loaders.py` | Half of it asserts what must **not** be removed: `_strip_table_lines` deletes text from the corpus, and over-deleting is unrecoverable and invisible. |
| `test_pipeline.py` | Driving `/ask` with PII in the question, neither the toolbox nor the router is ever constructed, and no message is persisted. |

Two of them pin couplings that would otherwise be invisible. `_page_based_chunking` does not
write `chunk_type` — parents are marked as a side effect of generating the micro chunks —
and since retrieval filters on that field and defaults to `micro`, skipping micro generation
would match nothing and return zero results without an error anywhere.

---

## Running on two clouds

The provider and the host are independent choices. Either can change without the other.

| | Railway | AWS |
|---|---|---|
| Compute | Container, always on | Lambda container behind API Gateway |
| Generation | OpenAI `gpt-4o-mini` | **Bedrock** — Claude Haiku 4.5 |
| Embeddings | OpenAI `text-embedding-3-small` | OpenAI (unchanged — see below) |
| Application database | PostgreSQL | RDS PostgreSQL |
| Entry point | gunicorn (`startup.sh`) | `lambda_handler.py` via `apig-wsgi` |

Switching provider is one environment variable:

```bash
LLM_PROVIDER=openai    # default
LLM_PROVIDER=bedrock
```

`app/rag_logic/llm_factory.py` is the only file in the repository that names a provider.
Everything else asks for a model and does not know who serves it.

**Embeddings deliberately stay on OpenAI.** The vectors in Chroma were built with
`text-embedding-3-small`; changing the embedding model invalidates the whole index, forces
a full re-ingest, and invalidates the evaluation numbers above.

### What Lambda demands that a container host does not

Three failures share one root: `/var/task` is read-only and only `/tmp` can be written.

1. **A Chroma vector store cannot be served read-only from the image.** The usual advice is
   to bake it in and read it in place. Chroma opens its SQLite read-write even for queries
   because it needs its journal, so retrieval fails with `attempt to write a readonly
   database (code: 8)` while the index sits right there. It is copied to `/tmp` during the
   init phase — 8.7 MB, once per container.
2. **Flask's `instance_path` lives next to the code**, so anything writing there returns 500
   until it is pointed at `/tmp`.
3. **Lambda rejects OCI image manifests.** It needs Docker v2 schema 2, and
   `--provenance=false` alone is not enough — the push must set `oci-mediatypes=false`.

Memory is over-allocated on purpose — in Lambda CPU scales with memory, and lowering it
lengthens the cold start.

### Measured on both deployments

| | Railway | AWS Lambda |
|---|---|---|
| Generation model | OpenAI `gpt-4o-mini` | Claude Haiku 4.5 |
| `/health`, warm | 0.46 s | 0.21 s |
| `/ask`, warm | 6.3 s | 8.6 - 11.2 s |
| First `/ask` on a cold container | 21.4 s | 16.6 - 26.5 s (four runs) |
| Request ceiling | none | 29 s (API Gateway) |

The function had been answering with Claude Sonnet, because `MODEL_NAME` was never set on
it and `config.py` defaults to `gpt-4o`, which the factory maps to Sonnet. Three to four
sequential calls to a large model took a cold request past 35 s, and API Gateway returned
503 at its ceiling — CloudWatch recorded `Duration: 35654 ms` against a request the caller
saw fail, because the function itself always completed. The model is now declared in
`Dockerfile.lambda` rather than inherited: the default in `config.py` is written for
Railway, where no request ceiling exists.

The cold figure is a range because it is one: four consecutive cold containers answered the
same question in 16.6, 17.1, 18.1 and 26.5 seconds. None returned 503, and the slowest came
within 2.5 s of the ceiling.

---

## Honest limitations

- **The 29 s API Gateway ceiling is a margin, not headroom.** A cold `/ask` measured
  between 16.6 s and 26.5 s across four runs — none failed, and the slowest was 2.5 s away.
  Roughly 8 s of a cold request is building the chain, and moving that into the init phase
  is the next lever; streaming is what removes the ceiling rather than widening it. Railway
  has no such ceiling.
- **CI, not CD.** `.github/workflows/ci.yml` runs ruff, pytest and a Docker build with
  `push: false`. There is no deploy job; Railway deploys from its own git integration,
  gated on Actions passing.
- **Not every module is covered.** 295 tests cover the router, the guardrails, ingestion,
  the retrieval decisions, the models and the routes. The console logger, the PowerPoint
  loader and the summariser have none.
- **Whole-word matching in the router loses morphological variants.** The heuristics used
  to match substrings, so `"suma"` matched inside `"consumar"` and routed to the wrong tool
  with no LLM to catch it. They now match whole words with accents normalised, which costs
  the other direction: `"empleado"` no longer matches the keyword `"empleados"`, so the
  lists spell out the forms that matter. It is the right trade for this component — a false
  negative reaches the LLM, which decides well; a false positive reaches the wrong tool with
  nothing behind it.
- **BM25 tokenises on whitespace only** — no stemming, no accent normalisation. In Spanish
  `"política"` and `"politica"` are different terms.
- Chunking is **recursive by tokens**, not semantic. The semantic part is an LLM enrichment
  pass at ingest time that prepends a headline and a summary before embedding.
- `evaluate_rag.py` builds its own retriever, so its numbers describe that configuration
  rather than the deployed chain. `evaluate_pipeline.py` exists for the deployed chain.

---

## Quick start

```bash
git clone https://github.com/sixtorapa/RAG_HR_ASSISTANT.git
cd RAG_HR_ASSISTANT

python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-prod.txt

printf 'OPENAI_API_KEY=your-key\n' > .env

python seed_hr_db.py                                    # toy HR SQLite database
KNOWLEDGE_BASE_PATH=$PWD/knowledge_base python ingest.py --force
pytest                                                  # 295 tests; OpenAI is mocked
python run.py                                           # http://localhost:5001
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

Model IDs are **inference profiles** (`eu.anthropic.claude-sonnet-4-6`), not bare model IDs.
The `eu.` prefix keeps inference inside the EU, which matters for a system holding employee
data. A bare `anthropic.claude-…` returns `ValidationException`.

---

## Observability

`observability.py` enables LangSmith tracing from environment variables and is called at
startup. `cost_calculator.py` computes per-query cost from token usage, and reports an
unpriced model as an error rather than silently counting it as free.

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key
LANGCHAIN_PROJECT=hr-kb-assistant
```

---

## Project structure

```
├── run.py                      # Flask entry point (Railway)
├── lambda_handler.py           # Lambda entry point (apig-wsgi)
├── config.py                   # All configuration via env vars
├── Dockerfile                  # Railway image
├── Dockerfile.lambda           # Lambda image (AWS base, vector store baked in)
│
├── app/main/
│   ├── routes.py               # The chat endpoints
│   ├── pipeline.py             # Toolbox, dispatch loop, answer flow
│   ├── guards.py               # Daily quota and input-side DLP
│   ├── views.py                # HTML screens
│   ├── chats.py                # Chat sessions and re-indexing
│   ├── auth.py                 # Login / logout
│   └── admin.py                # Activity panel and CSV export
│
├── app/rag_logic/
│   ├── llm_factory.py          # ← the only file that names a provider
│   ├── agent_router.py         # Three-path routing
│   ├── qa_chain.py             # RBAC, prefilter, two-pass, hybrid, parent expansion
│   ├── ingester.py             # Loaders, chunking, LLM enrichment, indexing
│   ├── custom_loaders.py       # PDF/PPTX loaders, layout-aware text and tables
│   ├── bm25_index.py           # BM25 built at ingest, persisted, loaded at query
│   ├── sql_tool.py             # Text-to-SQL with guardrails
│   ├── pii_guard.py            # Checksum-based PII detection
│   └── cost_calculator.py      # Per-query cost from token usage
│
├── evaluation/
│   ├── evaluate_pipeline.py    # RAGAS over the deployed chain
│   └── evaluate_rag.py         # RAGAS over three retrieval configurations
├── infra/                      # How the AWS side was built, in order
└── tests/                      # 295 tests, OpenAI mocked
```

### infra/

The seven scripts that created the AWS deployment, numbered in the order they were run:
the IAM execution role, the ECR repository and image push, the Lambda function, the API
Gateway endpoint, the RDS instance, and the schema initialisation.

They are honest about what they are: **imperative scripts, not infrastructure as code**.
There is no state file, no plan, and nothing detects drift. `infra/README.md` says so
explicitly, along with the security trade-off behind keeping the Lambda outside the VPC and
the teardown commands.

---

## License

MIT — see [LICENSE](LICENSE).

The knowledge base under `knowledge_base/` and the HR database are **synthetic**, written
for this project. There is no real employee data anywhere in the repository or behind the
public demo.
