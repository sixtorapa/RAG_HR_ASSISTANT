"""
evaluate_pipeline.py — evalúa el pipeline QUE SE DESPLIEGA, no una copia de él.

Por qué existe
--------------
`evaluate_rag.py` compara tres configuraciones de retrieval, y es útil para eso,
pero construye su propio retriever: k=5, sin two-pass, sin guardarraíl de acceso,
con su propio prompt de respuesta. Cero referencias a
`get_conversational_qa_chain`, a la capa de agentes o al router. Es decir: sus
números describen una reimplementación paralela, no el sistema que responde en
producción con k_base=28, dos pasadas y ACL.

Este script llama a la cadena real y mide el efecto de UNA capa concreta:

Mide UNA configuración del pipeline real y guarda el resultado etiquetado con
la granularidad activa (RETRIEVAL_CHUNK_TYPE), para poder comparar tiradas.

Uso:
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
MAX_CONTEXTOS = 10   # tope solo para el juez: RAGAS con 40 chunks infla el recall
                     # y dispara el gasto. Es IDÉNTICO en las dos variantes, así
                     # que no puede sesgar la comparación.
SALIDA = REPO_ROOT / "evaluation" / "pipeline_results.json"


def _contextos(docs) -> list[str]:
    return [d.page_content for d in (docs or []) if getattr(d, "page_content", "")][:MAX_CONTEXTOS]


def main() -> None:
    preguntas = [q for q in GOLDEN_DATASET if q["category"] == "rag"]
    print(f"▶ {len(preguntas)} preguntas de RAG del golden dataset")
    print(f"▶ modelo: {MODELO} · proveedor: {os.environ.get('LLM_PROVIDER', 'openai')}\n")

    app = create_app(Config)
    with app.app_context():
        vector_store_path = app.config["UP_VECTOR_STORE_PATH"]
        print(f"▶ vector store: {vector_store_path}\n")

        # allowed_departments=None -> sin restricción (equivale a un admin). Se fija
        # a propósito: si el usuario de prueba tuviera ACL parcial, la mitad de las
        # preguntas devolvería cero resultados y las métricas medirían el RBAC,
        # no la calidad de la respuesta.
        ajustes = {"allowed_departments": None, "k_base": 28}
        grano = os.environ.get("RETRIEVAL_CHUNK_TYPE", "micro")

        registros = []
        t0 = time.time()

        with get_openai_callback() as cb:
            for i, q in enumerate(preguntas, 1):
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
                print(f"  [{i:2}/{len(preguntas)}] {q['id']:8} {dt:5.1f}s · "
                      f"{len(docs):3} chunks · {chars:6,} chars")

        segundos = time.time() - t0
        print(f"\n▶ generación terminada en {segundos:.0f}s · "
              f"{cb.prompt_tokens:,} tokens de prompt · {cb.completion_tokens:,} de completion "
              f"· ${cb.total_cost:.4f}\n")

        print("▶ puntuando con RAGAS (LLM-as-judge)...")
        juez = get_llm(MODELO, 0.0)
        embeddings = get_embeddings()
        build_ragas_metrics(juez, embeddings)
        scores = run_ragas_evaluation(registros, juez, embeddings)

        # Línea base: tirada del 3-ago-2026 SIN filtro de granularidad, sobre la
        # misma cadena y el mismo golden dataset.
        BASE = {"context_precision": 0.7220, "context_recall": 0.8095,
                "faithfulness": 0.9382, "answer_relevancy": 0.8327}

        print("\n" + "=" * 78)
        print(f"granularidad activa: RETRIEVAL_CHUNK_TYPE={grano}")
        print(f"{'métrica':<22}{'base (all)':>14}{'ahora':>14}{'delta':>14}")
        print("-" * 78)
        for m in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            a, b = BASE[m], scores.get(m)
            if b is None:
                print(f"{m:<22}{a:>14.4f}{'—':>14}{'—':>14}")
                continue
            print(f"{m:<22}{a:>14.4f}{b:>14.4f}{b - a:>+14.4f}")
        print("=" * 78)
        print("\nRecuerda el suelo de ruido del juez medido en la tirada pareada: +-0.004.")

        SALIDA.write_text(json.dumps({
            "fecha": datetime.now().isoformat(),
            "modelo": MODELO,
            "n_preguntas": len(preguntas),
            "max_contextos_para_el_juez": MAX_CONTEXTOS,
            "segundos_generacion": round(segundos, 1),
            "coste_generacion_usd": round(cb.total_cost, 4),
            "granularidad": grano,
            "resultados": scores,
            "linea_base_all": BASE,
            "nota": (
                "Pipeline real (get_conversational_qa_chain con two-pass, ACL e hibrido). "
                "La linea base es la tirada del 3-ago-2026 sin filtro de granularidad."
            ),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ guardado en {SALIDA}")


if __name__ == "__main__":
    main()
