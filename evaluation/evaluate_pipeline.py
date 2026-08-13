"""
evaluate_pipeline.py — evaluates the pipeline THAT IS DEPLOYED, not a copy of it.

Why it exists
-------------
`evaluate_rag.py` compares three retrieval configurations, which is useful for
that purpose, but it builds its own retriever: k=5, no two-pass, no access
guardrail, its own answer prompt. Zero references to
`get_conversational_qa_chain`, to the agent layer or to the router. Its numbers
therefore describe a parallel reimplementation, not the system that answers in
production with k_base=28, two passes and an ACL.

This script calls the real chain, measures ONE configuration of it, and stores
the result tagged with the active granularity (RETRIEVAL_CHUNK_TYPE) so runs can
be compared.

Usage:
    .venv/bin/python -m evaluation.evaluate_pipeline
    RETRIEVAL_CHUNK_TYPE=all .venv/bin/python -m evaluation.evaluate_pipeline
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import create_app                                    # noqa: E402
from app.rag_logic.llm_factory import get_embeddings, get_llm  # noqa: E402
from app.rag_logic.tools import ChatWithDocumentTool          # noqa: E402
from config import Config                                     # noqa: E402
from evaluation.evaluate_rag import (                         # noqa: E402
    GOLDEN_DATASET,
    build_ragas_metrics,
    run_ragas_evaluation,
)
from langchain_community.callbacks import get_openai_callback  # noqa: E402

MODELO = "gpt-4o-mini"
MAX_CONTEXTOS = 10   # cap for the judge only: RAGAS with 40 chunks inflates recall
                     # and drives the spend. It is IDENTICAL in both variants, so it
                     # cannot bias the comparison.
SALIDA = REPO_ROOT / "evaluation" / "pipeline_results.json"


def _contextos(docs) -> list[str]:
    return [d.page_content for d in (docs or []) if getattr(d, "page_content", "")][:MAX_CONTEXTOS]


def main() -> None:
    questions = [q for q in GOLDEN_DATASET if q["category"] == "rag"]
    print(f"▶ {len(questions)} RAG questions from the golden dataset")
    print(f"▶ modelo: {MODELO} · proveedor: {os.environ.get('LLM_PROVIDER', 'openai')}\n")

    app = create_app(Config)
    with app.app_context():
        vector_store_path = app.config["UP_VECTOR_STORE_PATH"]
        print(f"▶ vector store: {vector_store_path}\n")

        # allowed_departments=None -> unrestricted (equivalent to an admin). Set
        # deliberately: with a partial ACL on the test user, half the questions
        # would return zero results and the metrics would be measuring the RBAC,
        # not the quality of the answer.
        ajustes = {"allowed_departments": None, "k_base": 28}
        grano = os.environ.get("RETRIEVAL_CHUNK_TYPE", "micro")

        registros = []
        t0 = time.time()

        with get_openai_callback() as cb:
            for i, q in enumerate(questions, 1):
                pregunta = q["question"]
                tool = ChatWithDocumentTool(
                    project_id="eval",
                    vector_store_path=vector_store_path,
                    model_name=MODELO,
                    project_settings=dict(ajustes, last_user_question=pregunta),
                )

                t_ini = time.time()
                base = tool.run({"question": pregunta, "chat_history": []})
                if not isinstance(base, dict):
                    base = {"answer": str(base), "source_documents": []}
                dt = time.time() - t_ini

                docs = base.get("source_documents", []) or []
                registros.append({
                    "question": pregunta,
                    "answer": (base.get("answer") or "").strip(),
                    "contexts": _contextos(docs),
                    "ground_truth": q["ground_truth"],
                })
                chars = sum(len(d.page_content) for d in docs)
                print(f"  [{i:2}/{len(questions)}] {q['id']:8} {dt:5.1f}s · "
                      f"{len(docs):3} chunks · {chars:6,} chars")

        seconds = time.time() - t0
        print(f"\n▶ generation finished in {seconds:.0f}s · "
              f"{cb.prompt_tokens:,} prompt tokens · {cb.completion_tokens:,} completion "
              f"· ${cb.total_cost:.4f}\n")

        print("▶ scoring with RAGAS (LLM-as-judge)...")
        juez = get_llm(MODELO, 0.0)
        embeddings = get_embeddings()
        build_ragas_metrics(juez, embeddings)
        scores = run_ragas_evaluation(registros, juez, embeddings)

        # Baseline: a run WITHOUT the granularity filter, over the same chain
        # and the same golden dataset.
        BASE = {"context_precision": 0.7220, "context_recall": 0.8095,
                "faithfulness": 0.9382, "answer_relevancy": 0.8327}

        print("\n" + "=" * 78)
        print(f"granularidad activa: RETRIEVAL_CHUNK_TYPE={grano}")
        print(f"{'metric':<22}{'base (all)':>14}{'now':>14}{'delta':>14}")
        print("-" * 78)
        for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            a, b = BASE[m], scores.get(m)
            if b is None:
                print(f"{m:<22}{a:>14.4f}{'—':>14}{'—':>14}")
                continue
            print(f"{m:<22}{a:>14.4f}{b:>14.4f}{b - a:>+14.4f}")
        print("=" * 78)
        print("\nNote the judge noise floor measured on the paired run: +-0.004.")

        SALIDA.write_text(json.dumps({
            "fecha": datetime.now().isoformat(),
            "modelo": MODELO,
            "n_preguntas": len(questions),
            "max_contextos_para_el_juez": MAX_CONTEXTOS,
            "segundos_generacion": round(seconds, 1),
            "coste_generacion_usd": round(cb.total_cost, 4),
            "granularidad": grano,
            "resultados": scores,
            "linea_base_all": BASE,
            "nota": (
                "Pipeline real (get_conversational_qa_chain con two-pass, ACL e hibrido). "
                "La linea base es la tirada del 3-ago-2026 sin filtro de granularidad."
            ),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ saved to {SALIDA}")


if __name__ == "__main__":
    main()
