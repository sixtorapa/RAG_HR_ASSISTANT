"""
evaluate_rag.py — RAGAS Evaluation Suite for HR Knowledge Base Assistant
=========================================================================

Compares three retrieval pipeline configurations on a 25-question golden
dataset built from the 5 HR policy documents ingested into ChromaDB.

Configurations tested:
  1. Vector only          — MMR vector search (baseline)
  2. BM25 + Vector        — EnsembleRetriever 55/45 RRF  ← PRODUCTION (Railway)
  3. BM25 + Vector + Rerank — + FlashRank reranking       (local only, OOM on Railway)

RAGAS metrics (ragas 0.2.x):
  context_precision   — What fraction of retrieved chunks are actually relevant?
  context_recall      — Does the retrieved context contain the full answer?
  faithfulness        — Is the answer grounded in the context (no hallucination)?
  answer_relevancy    — Does the answer address the question asked?

Requirements:
    pip install "ragas>=0.2.0,<0.3.0" --prefer-binary

Usage:
    # Full evaluation (all 3 configs):
    python evaluate_rag.py --skip-sql

    # Match Railway production (no FlashRank):
    python evaluate_rag.py --skip-sql --no-rerank

    # Inspect golden dataset without API calls:
    python evaluate_rag.py --list-questions

    # Custom vector store path:
    python evaluate_rag.py --vector-store /path/to/vector_store/info

Output:
    eval_results.json  — full per-question results for all configs
    eval_results.csv   — summary scores table (copy-paste for README)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Silence noisy loggers before any imports ────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for _noisy in ("chromadb", "langchain", "langchain_core", "httpx", "openai",
               "httpcore", "urllib3", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Load env + sanitize API key ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

raw_key = os.environ.get("OPENAI_API_KEY", "")
if raw_key:
    clean_key = raw_key.strip().strip("'").strip('"')
    os.environ["OPENAI_API_KEY"] = clean_key
else:
    print("❌  OPENAI_API_KEY is not set. Add it to your .env file.")
    sys.exit(1)

# ── RAGAS 0.2.x imports ──────────────────────────────────────────────────────
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    print("⚠️  RAGAS not installed.")
    print("   Run: pip install 'ragas>=0.2.0,<0.3.0' --prefer-binary\n")

# ── LangChain imports ────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

try:
    from langchain.retrievers.document_compressors import FlashrankRerank
    _FLASHRANK_AVAILABLE = True
except Exception:
    _FLASHRANK_AVAILABLE = False

from config import Config


# ═══════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 25 questions from 5 HR documents + SQL database
#
# ground_truth: exact answer used by context_recall to check if retrieved
#               chunks contain the information needed to answer correctly.
# source_doc:   which document the answer lives in (for per-doc analysis).
# category:     "rag" (document retrieval) | "sql" (database queries)
# ═══════════════════════════════════════════════════════════════════════════

GOLDEN_DATASET: list[dict] = [

    # ── 01_employee_handbook ────────────────────────────────────────────────
    {
        "id": "EH-01",
        "question": "How many days of annual leave are employees entitled to?",
        "ground_truth": "Employees are entitled to 25 days of annual leave per year.",
        "source_doc": "01_employee_handbook",
        "category": "rag",
    },
    {
        "id": "EH-02",
        "question": "How many days per week can employees work remotely?",
        "ground_truth": (
            "Employees may work remotely up to 3 days per week after completing "
            "their probation period of 90 days. Full-remote arrangements require VP approval."
        ),
        "source_doc": "01_employee_handbook",
        "category": "rag",
    },
    {
        "id": "EH-03",
        "question": "What is the home office allowance?",
        "ground_truth": (
            "Employees receive a €500 one-time setup allowance plus €30 per month "
            "for home office expenses."
        ),
        "source_doc": "01_employee_handbook",
        "category": "rag",
    },
    {
        "id": "EH-04",
        "question": "How many free therapy sessions does the company provide per year?",
        "ground_truth": (
            "The company provides 8 free therapy sessions per year via Oliva Health "
            "as part of the mental health support benefit."
        ),
        "source_doc": "01_employee_handbook",
        "category": "rag",
    },
    {
        "id": "EH-05",
        "question": "What is the parental leave entitlement for primary and secondary caregivers?",
        "ground_truth": (
            "Primary caregivers receive 16 weeks of parental leave. "
            "Secondary caregivers receive 6 weeks. Notice required is 8 weeks minimum."
        ),
        "source_doc": "01_employee_handbook",
        "category": "rag",
    },

    # ── 02_performance_review_guide ──────────────────────────────────────────
    {
        "id": "PR-01",
        "question": "When does the self-assessment phase open for the annual review?",
        "ground_truth": (
            "The self-assessment phase opens on 1 November and closes on 15 November. "
            "Manager reviews follow from 16 to 30 November."
        ),
        "source_doc": "02_performance_review_guide",
        "category": "rag",
    },
    {
        "id": "PR-02",
        "question": "What score is required to be eligible for a promotion?",
        "ground_truth": (
            "A score of 4.0 or above in the last review is required for promotion, "
            "along with a minimum of 12 months in the current role and manager and VP endorsement."
        ),
        "source_doc": "02_performance_review_guide",
        "category": "rag",
    },
    {
        "id": "PR-03",
        "question": "What percentage of employees can receive an Outstanding rating per department?",
        "ground_truth": (
            "A maximum of 15% of employees per department can receive an Outstanding rating "
            "during calibration."
        ),
        "source_doc": "02_performance_review_guide",
        "category": "rag",
    },
    {
        "id": "PR-04",
        "question": "What are the four dimensions used to evaluate performance and their weights?",
        "ground_truth": (
            "The four dimensions are: Delivery & Results (35%), Collaboration & Teamwork (25%), "
            "Growth & Learning (20%), and Leadership & Initiative (20%)."
        ),
        "source_doc": "02_performance_review_guide",
        "category": "rag",
    },

    # ── 03_onboarding_guide ─────────────────────────────────────────────────
    {
        "id": "OB-01",
        "question": "How many days before the start date does HR send the welcome pack?",
        "ground_truth": (
            "HR sends the welcome pack 5 business days before the employee's start date. "
            "It includes the employment contract, GDPR form, IT equipment confirmation, "
            "and access credentials."
        ),
        "source_doc": "03_onboarding_guide",
        "category": "rag",
    },
    {
        "id": "OB-02",
        "question": "What is the probation period and what is not available during it?",
        "ground_truth": (
            "The probation period lasts 90 days. During this time remote working "
            "and stock options are not available."
        ),
        "source_doc": "03_onboarding_guide",
        "category": "rag",
    },
    {
        "id": "OB-03",
        "question": "What should a new employee complete in their first 30 days?",
        "ground_truth": (
            "In the first 30 days employees should complete all mandatory training "
            "(GDPR, Security, Anti-bribery), shadow at least 3 team members, set up 1:1s "
            "with key stakeholders, and understand the team's OKRs for the current quarter."
        ),
        "source_doc": "03_onboarding_guide",
        "category": "rag",
    },
    {
        "id": "OB-04",
        "question": "What tool is used for performance management and OKRs?",
        "ground_truth": (
            "Lattice is the tool used for performance management and OKRs. "
            "Access is provided by HR during the first week of onboarding."
        ),
        "source_doc": "03_onboarding_guide",
        "category": "rag",
    },

    # ── 04_compensation_and_benefits ────────────────────────────────────────
    {
        "id": "CB-01",
        "question": "What is the salary band for a Senior level employee?",
        "ground_truth": "The salary band for Senior level employees is €55,000 to €80,000 per year.",
        "source_doc": "04_compensation_and_benefits",
        "category": "rag",
    },
    {
        "id": "CB-02",
        "question": "What is the bonus target percentage for a Lead level employee?",
        "ground_truth": (
            "The bonus target for Lead level employees is 15% of base salary. "
            "Actual payout ranges from 0% to 2x target depending on individual and company performance."
        ),
        "source_doc": "04_compensation_and_benefits",
        "category": "rag",
    },
    {
        "id": "CB-03",
        "question": "How does the stock option vesting schedule work?",
        "ground_truth": (
            "Stock options vest over 4 years with a 1-year cliff: 25% vests after year 1, "
            "and the remaining 75% vests monthly over years 2 to 4. "
            "The exercise window is 90 days after leaving the company."
        ),
        "source_doc": "04_compensation_and_benefits",
        "category": "rag",
    },
    {
        "id": "CB-04",
        "question": "What pay increase can an employee with an Outstanding score expect?",
        "ground_truth": (
            "An employee with an Outstanding performance score (4.5–5.0) can expect "
            "a pay increase of 6 to 10%."
        ),
        "source_doc": "04_compensation_and_benefits",
        "category": "rag",
    },
    {
        "id": "CB-05",
        "question": "What is the meal allowance and transport benefit?",
        "ground_truth": (
            "Employees receive €8 per working day as a meal allowance via a company card, "
            "and €60 per month for a public transport card."
        ),
        "source_doc": "04_compensation_and_benefits",
        "category": "rag",
    },

    # ── 05_recruitment_and_hiring_policy ────────────────────────────────────
    {
        "id": "RH-01",
        "question": "How long does the full interview process typically take from JR approval to offer?",
        "ground_truth": (
            "The full interview process takes approximately 16 to 18 business days from "
            "JR approval to written offer, based on the stage SLAs defined in the policy."
        ),
        "source_doc": "05_recruitment_and_hiring_policy",
        "category": "rag",
    },
    {
        "id": "RH-02",
        "question": "What is the referral bonus and when is it paid?",
        "ground_truth": (
            "Employees receive a €1,000 cash bonus for a successful referral, paid "
            "in the next payroll after the referred candidate completes 3 months. "
            "Referrals within the same reporting line are not allowed."
        ),
        "source_doc": "05_recruitment_and_hiring_policy",
        "category": "rag",
    },
    {
        "id": "RH-03",
        "question": "What topics are prohibited in job interviews?",
        "ground_truth": (
            "Interviewers must never ask about age, date of birth, graduation year, "
            "marital status, family plans, nationality, country of origin, religious beliefs, "
            "political views, or disability and health conditions before an offer."
        ),
        "source_doc": "05_recruitment_and_hiring_policy",
        "category": "rag",
    },

    # ── SQL / HR Database ────────────────────────────────────────────────────
    {
        "id": "SQL-01",
        "question": "How many departments are there in the company?",
        "ground_truth": (
            "There are 6 departments: Engineering, HR, Sales, Finance, Marketing, and Product."
        ),
        "source_doc": "sql_database",
        "category": "sql",
    },
    {
        "id": "SQL-02",
        "question": "What locations do employees work from?",
        "ground_truth": (
            "Employees work from four locations: Madrid, Barcelona, Remote, and London."
        ),
        "source_doc": "sql_database",
        "category": "sql",
    },
    {
        "id": "SQL-03",
        "question": "What is the budget of the Engineering department?",
        "ground_truth": "The Engineering department has a budget of €850,000.",
        "source_doc": "sql_database",
        "category": "sql",
    },
    {
        "id": "SQL-04",
        "question": "What job postings are currently open?",
        "ground_truth": (
            "Currently open positions include: Senior Software Engineer (Engineering), "
            "HR Business Partner (HR), Account Executive EMEA (Sales), "
            "Growth Marketing Manager (Marketing), and DevOps Engineer (Engineering)."
        ),
        "source_doc": "sql_database",
        "category": "sql",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVER FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_retriever(vector_store_path: str, config_name: str, k: int = 5):
    """
    Build a retriever for the given pipeline configuration.

    Configurations:
        "vector_only"   — MMR vector search only (baseline)
        "bm25_vector"   — BM25 + vector EnsembleRetriever 55/45 RRF
                          *** PRODUCTION config on Railway ***
                          (FlashRank disabled there due to OOM on free tier)
        "full_pipeline" — BM25 + vector + FlashRank reranking
                          Local only. Use --no-rerank to skip.

    k controls how many final chunks are returned. Internally fetches k*5
    candidates to give BM25 and rerank room to improve ordering.

    Returns (retriever, vector_store).
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embeddings,
    )

    fetch_k = k * 5  # retrieve more candidates, then filter down

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": fetch_k, "fetch_k": fetch_k * 3, "lambda_mult": 0.55},
    )

    if config_name == "vector_only":
        return vector_retriever, vector_store

    # ── Load all chunks for BM25 ──────────────────────────────────────────────
    data = vector_store.get(include=["documents", "metadatas"])
    docs_text = data.get("documents", []) or []
    docs_meta = data.get("metadatas", []) or []

    if not docs_text:
        print("❌  Vector store is empty — run python ingest.py first.")
        sys.exit(1)

    docs_for_bm25 = [
        Document(page_content=t, metadata=(m or {}))
        for t, m in zip(docs_text, docs_meta)
    ]
    bm25 = BM25Retriever.from_documents(docs_for_bm25)
    bm25.k = fetch_k

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.55, 0.45],
    )

    if config_name == "bm25_vector":
        return ensemble, vector_store

    # ── FlashRank reranking (local only) ─────────────────────────────────────
    if not _FLASHRANK_AVAILABLE:
        print("⚠️  FlashRank not available — falling back to bm25_vector.")
        return ensemble, vector_store

    try:
        compressor = FlashrankRerank(top_n=k)
        reranked = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble,
        )
        return reranked, vector_store
    except Exception as e:
        print(f"⚠️  FlashRank init failed ({e}) — falling back to bm25_vector.")
        return ensemble, vector_store


# ═══════════════════════════════════════════════════════════════════════════
# ANSWER GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

ANSWER_PROMPT = """\
You are an HR assistant. Answer the question using ONLY the provided context.
Be concise and accurate. If the context does not contain the answer, say "Not found in context."

Context:
{context}

Question: {question}

Answer:"""


def generate_answer_and_contexts(
    question: str,
    retriever,
    llm: ChatOpenAI,
    k: int = 5,
) -> tuple[str, list[str]]:
    """
    Retrieve context chunks and generate an LLM answer for one question.

    Contexts are capped at k before being sent to the LLM and to RAGAS.
    Sending 20-30 chunks inflates recall trivially and causes TPM rate limits.

    Returns (answer_text, context_list).
    """
    try:
        docs = retriever.invoke(question)
    except Exception as e:
        print(f"  ⚠️  Retrieval error: {e}")
        return "Retrieval failed.", []

    docs = docs[:k]
    contexts = [doc.page_content for doc in docs if doc.page_content]

    if not contexts:
        return "No relevant context found.", []

    context_str = "\n\n---\n\n".join(contexts)
    prompt = ANSWER_PROMPT.format(context=context_str, question=question)

    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        answer = f"LLM error: {e}"

    return answer, contexts


# ═══════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def build_ragas_metrics(llm: ChatOpenAI, embeddings: OpenAIEmbeddings) -> list:
    """
    Instantiate RAGAS 0.2.x metrics with explicit LLM and embeddings wrappers.

    ResponseRelevancy requires both LLM and embeddings to be passed explicitly.
    Without this, it silently returns 0.0 (a known 0.2.x gotcha).
    """
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    return [
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]


def run_ragas_evaluation(
    eval_data: list[dict],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
) -> dict[str, float | None]:
    """
    Score a set of QA results using RAGAS 0.2.x.

    Returns dict: metric_name -> float score (or None on failure).
    """
    _null = {m: None for m in ["context_precision", "context_recall",
                                "faithfulness", "answer_relevancy"]}

    if not _RAGAS_AVAILABLE:
        print("  ⚠️  RAGAS not available — skipping.")
        return _null

    valid = [item for item in eval_data if item.get("contexts")]
    if not valid:
        print("  ⚠️  No valid contexts to evaluate.")
        return _null

    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            reference=item["ground_truth"],
        )
        for item in valid
    ]
    dataset = EvaluationDataset(samples=samples)

    try:
        result = ragas_evaluate(dataset=dataset, metrics=build_ragas_metrics(llm, embeddings))
        df = result.to_pandas().mean(numeric_only=True)
        return {
            "context_precision": round(float(df.get("llm_context_precision_with_reference", 0)), 4),
            "context_recall":    round(float(df.get("context_recall", 0)), 4),
            "faithfulness":      round(float(df.get("faithfulness", 0)), 4),
            "answer_relevancy":  round(float(df.get("answer_relevancy", 0)), 4),
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS error: {e}")
        return _null


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGURATION REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

ALL_CONFIGS = ["vector_only", "bm25_vector", "full_pipeline"]

CONFIG_LABELS = {
    "vector_only":   "Vector only (MMR)",
    "bm25_vector":   "BM25 + Vector  [PRODUCTION]",
    "full_pipeline": "BM25 + Vector + Rerank",
}

METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

# Rate-limit protection
SLEEP_BETWEEN_QUESTIONS = 1.5   # seconds between each question
SLEEP_BETWEEN_CONFIGS   = 30    # seconds between configurations


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    vector_store_path: str,
    skip_sql: bool = False,
    no_rerank: bool = False,
    output_path: str = "eval_results.json",
    k: int = 5,
) -> None:
    """
    Run the full evaluation loop across all pipeline configurations.

    Args:
        vector_store_path: Path to ChromaDB persist directory.
        skip_sql:          Exclude SQL questions from the golden dataset.
        no_rerank:         Skip full_pipeline config (matches Railway production).
        output_path:       Where to write JSON results.
        k:                 Number of final context chunks per query.
    """
    if not Path(vector_store_path).exists():
        print(f"❌  Vector store not found at: {vector_store_path}")
        print("    Run: python ingest.py first.")
        sys.exit(1)

    # Decide which configs to run
    configs_to_run = ["vector_only", "bm25_vector"]
    if not no_rerank:
        if _FLASHRANK_AVAILABLE:
            configs_to_run.append("full_pipeline")
        else:
            print("⚠️  FlashRank not installed — skipping full_pipeline.\n")

    # Filter golden dataset
    questions     = [q for q in GOLDEN_DATASET if not (skip_sql and q["category"] == "sql")]
    rag_questions = [q for q in questions if q["category"] == "rag"]

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  HR Knowledge Base — RAG Evaluation Suite")
    print(f"  Date:           {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Vector store:   {vector_store_path}")
    print(f"  Questions:      {len(rag_questions)} RAG" +
          (f" + {len(questions) - len(rag_questions)} SQL" if not skip_sql else ""))
    print(f"  Configs:        {len(configs_to_run)}  ({', '.join(CONFIG_LABELS[c] for c in configs_to_run)})")
    print(f"  Contexts/query: {k} chunks (capped — avoids TPM rate limits in RAGAS)")
    print(f"  Production cfg: BM25 + Vector  (FlashRank OFF on Railway free tier)")
    print(f"  RAGAS:          {'✅ 0.2.x' if _RAGAS_AVAILABLE else '❌ not installed'}")
    if no_rerank:
        print(f"  Mode:           --no-rerank  (matches Railway production)")
    print("═" * 68 + "\n")

    # Shared clients — reused across configs and passed explicitly to RAGAS
    llm        = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    all_results: dict[str, Any] = {
        "evaluation_date":         datetime.now().isoformat(),
        "vector_store_path":       vector_store_path,
        "production_config":       "bm25_vector",
        "flashrank_in_production": False,
        "note": (
            "FlashRank is disabled in production (Railway free tier) due to OOM. "
            "bm25_vector is the active production pipeline."
        ),
        "total_rag_questions": len(rag_questions),
        "configurations":      {},
    }

    config_scores: dict[str, dict] = {}

    # ── Evaluate each configuration ──────────────────────────────────────────
    for i, config_name in enumerate(configs_to_run):
        label   = CONFIG_LABELS[config_name]
        is_prod = config_name == "bm25_vector"

        print(f"🔧 [{i+1}/{len(configs_to_run)}] {label}")
        print("-" * 55)

        retriever, _ = build_retriever(vector_store_path, config_name, k=k)

        eval_data: list[dict] = []
        for q in rag_questions:
            print(f"  [{q['id']}] {q['question'][:62]}...")
            t0 = time.time()
            answer, contexts = generate_answer_and_contexts(q["question"], retriever, llm, k=k)
            latency = round(time.time() - t0, 2)

            eval_data.append({
                "id":           q["id"],
                "question":     q["question"],
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": q["ground_truth"],
                "source_doc":   q["source_doc"],
                "latency_s":    latency,
            })
            print(f"    ✓ {latency}s | {len(contexts)} chunks")
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

        print(f"\n  📊 Scoring with RAGAS...")
        scores = run_ragas_evaluation(eval_data, llm, embeddings)
        config_scores[config_name] = scores

        all_results["configurations"][config_name] = {
            "label":         label,
            "is_production": is_prod,
            "scores":        scores,
            "per_question":  eval_data,
        }

        score_summary = " | ".join(
            f"{m.split('_')[0]}={v:.3f}" if v is not None else f"{m.split('_')[0]}=N/A"
            for m, v in scores.items()
        )
        print(f"  ✅ {score_summary}")

        if i < len(configs_to_run) - 1:
            print(f"\n  ⏳ Sleeping {SLEEP_BETWEEN_CONFIGS}s to recover TPM quota...\n")
            time.sleep(SLEEP_BETWEEN_CONFIGS)

    # ── Results table and delta summary ──────────────────────────────────────
    _print_results_table(config_scores, configs_to_run)

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Full results → {output_path}")

    csv_path = output_path.replace(".json", ".csv")
    _save_csv(config_scores, configs_to_run, csv_path)
    print(f"  📊 Summary CSV  → {csv_path}")
    print("\n" + "═" * 68 + "\n")


# ── Output helpers ────────────────────────────────────────────────────────

def _print_results_table(
    config_scores: dict[str, dict],
    configs_to_run: list[str],
) -> None:
    col_w = 26
    print("\n" + "═" * 68)
    print("  RESULTS — Pipeline Configuration Comparison")
    print("  [PRODUCTION] = active config on Railway (FlashRank disabled)")
    print("═" * 68)

    header = f"{'Metric':<24}"
    for cfg in configs_to_run:
        label = CONFIG_LABELS[cfg].replace(" [PRODUCTION]", " *")
        header += f"{label:<{col_w}}"
    print(header)
    print("-" * (24 + col_w * len(configs_to_run)))

    for metric in METRICS:
        row = f"{metric:<24}"
        for cfg in configs_to_run:
            val = config_scores.get(cfg, {}).get(metric)
            row += f"{val:<{col_w}.4f}" if val is not None else f"{'N/A':<{col_w}}"
        print(row)

    print("═" * 68)

    # Delta: production vs baseline
    if "bm25_vector" in config_scores and "vector_only" in config_scores:
        print("\n  📈 Production (BM25+Vector) vs Baseline (Vector only):")
        for m in METRICS:
            bv = config_scores["bm25_vector"].get(m)
            vo = config_scores["vector_only"].get(m)
            if bv is not None and vo is not None:
                delta = round(bv - vo, 4)
                sign  = "+" if delta >= 0 else ""
                print(f"     {m:<32} {sign}{delta:.4f}")

    # Delta: full pipeline vs baseline (only if it ran)
    if "full_pipeline" in config_scores and "vector_only" in config_scores:
        print("\n  📈 Full pipeline (local rerank) vs Baseline (Vector only):")
        for m in METRICS:
            fp = config_scores["full_pipeline"].get(m)
            vo = config_scores["vector_only"].get(m)
            if fp is not None and vo is not None:
                delta = round(fp - vo, 4)
                sign  = "+" if delta >= 0 else ""
                print(f"     {m:<32} {sign}{delta:.4f}")

    # Best config
    scored = [c for c in configs_to_run
              if any(config_scores.get(c, {}).get(m) is not None for m in METRICS)]
    if scored:
        best = max(scored, key=lambda c: sum(
            v for v in config_scores[c].values() if v is not None
        ))
        print(f"\n  🏆 Best overall: {CONFIG_LABELS[best]}")


def _save_csv(
    config_scores: dict[str, dict],
    configs_to_run: list[str],
    csv_path: str,
) -> None:
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["configuration"] + METRICS)
            for cfg in configs_to_run:
                row = [CONFIG_LABELS[cfg]]
                for m in METRICS:
                    val = config_scores.get(cfg, {}).get(m)
                    row.append(f"{val:.4f}" if val is not None else "N/A")
                writer.writerow(row)
    except Exception as e:
        print(f"  ⚠️  CSV save failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation suite for HR Knowledge Base RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_rag.py --skip-sql              # Full eval, RAG questions only
  python evaluate_rag.py --skip-sql --no-rerank  # Match Railway production exactly
  python evaluate_rag.py --list-questions        # Inspect dataset (no API calls)
        """,
    )
    parser.add_argument(
        "--vector-store",
        default=Config.UP_VECTOR_STORE_PATH,
        help="Path to ChromaDB persist directory (default: from config.py)",
    )
    parser.add_argument(
        "--skip-sql",
        action="store_true",
        help="Evaluate RAG questions only (skip SQL/database questions)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help=(
            "Skip full_pipeline (FlashRank) config. "
            "Matches Railway production where FlashRank is disabled due to OOM."
        ),
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Output path for full JSON results (default: eval_results.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of final context chunks per query (default: 5)",
    )
    parser.add_argument(
        "--list-questions",
        action="store_true",
        help="Print the golden dataset and exit (no API calls made)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_questions:
        print(f"\n{'ID':<8} {'Cat':<6} {'Source':<35} Question")
        print("─" * 95)
        for q in GOLDEN_DATASET:
            print(f"{q['id']:<8} {q['category']:<6} {q['source_doc']:<35} {q['question'][:48]}")
        rag_n = sum(1 for q in GOLDEN_DATASET if q["category"] == "rag")
        sql_n = sum(1 for q in GOLDEN_DATASET if q["category"] == "sql")
        print(f"\nTotal: {len(GOLDEN_DATASET)} questions ({rag_n} RAG, {sql_n} SQL)")
        sys.exit(0)

    run_evaluation(
        vector_store_path=args.vector_store,
        skip_sql=args.skip_sql,
        no_rerank=args.no_rerank,
        output_path=args.output,
        k=args.k,
    )