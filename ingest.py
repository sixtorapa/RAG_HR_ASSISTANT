"""
ingest.py — Knowledge Base Indexer
====================================
Run this script from the project root to index or re-index documents.

Usage:
    python ingest.py                  # index all new/changed files
    python ingest.py --force          # force full re-index of everything
    python ingest.py --docs ./path    # use a custom documents folder
    python ingest.py --vs ./path      # use a custom vector store path
"""

import os
import sys
import argparse
import time

# ── Make sure app imports work from project root ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Sanitize API key — strips quotes/whitespace that dotenv sometimes leaves in
raw_key = os.environ.get("OPENAI_API_KEY", "")
if raw_key:
    clean_key = raw_key.strip().strip("'").strip('"')
    os.environ["OPENAI_API_KEY"] = clean_key
else:
    print("❌  OPENAI_API_KEY is not set. Add it to your .env file.")
    sys.exit(1)

from config import Config
from app.rag_logic.ingester import process_and_store_documents


# ── Helpers ────────────────────────────────────────────────────────────────────

def count_files(folder: str) -> dict:
    """Count supported files in the knowledge base folder."""
    extensions = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".txt", ".md"}
    counts = {}
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                counts[ext] = counts.get(ext, 0) + 1
                total += 1
    return {"total": total, "by_type": counts}


def print_header():
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║       HR Knowledge Base — Document Indexer   ║")
    print("╚══════════════════════════════════════════════╝")
    print()


def print_summary(doc_path: str, vector_path: str):
    stats = count_files(doc_path)
    print(f"  📂  Knowledge base : {doc_path}")
    print(f"  🗄️   Vector store   : {vector_path}")
    print(f"  📄  Files found    : {stats['total']}", end="")
    if stats["by_type"]:
        breakdown = "  (" + ", ".join(f"{v} {k}" for k, v in sorted(stats["by_type"].items())) + ")"
        print(breakdown, end="")
    print()
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print_header()

    parser = argparse.ArgumentParser(
        description="Index documents into the HR Knowledge Base vector store.",
        add_help=True,
    )
    parser.add_argument(
        "--docs",
        default=None,
        help="Path to the documents folder (default: KNOWLEDGE_BASE_PATH from .env)",
    )
    parser.add_argument(
        "--vs",
        default=None,
        help="Path to the vector store (default: UP_VECTOR_STORE_PATH from .env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full re-index (ignore manifest, re-process all files)",
    )
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    doc_path    = args.docs or Config.KNOWLEDGE_BASE_PATH
    vector_path = args.vs   or Config.UP_VECTOR_STORE_PATH

    doc_path    = os.path.abspath(doc_path)
    vector_path = os.path.abspath(vector_path)

    # ── Validate ───────────────────────────────────────────────────────────────
    if not doc_path:
        print("❌  KNOWLEDGE_BASE_PATH is not set. Add it to your .env file.")
        sys.exit(1)

    if not os.path.exists(doc_path):
        print(f"❌  Documents folder not found: {doc_path}")
        print("    Create it and add your PDF / XLSX / DOCX files.")
        sys.exit(1)

    stats = count_files(doc_path)
    if stats["total"] == 0:
        print(f"⚠️   No supported documents found in: {doc_path}")
        print("    Supported types: .pdf  .docx  .pptx  .xlsx  .txt  .md")
        sys.exit(0)

    # ── Force mode: delete manifest so everything gets re-processed ───────────
    if args.force:
        manifest_path = os.path.join(vector_path, "_manifest.json")
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
            print("  🗑️   Manifest cleared — full re-index will run.\n")

    # ── Run ingestion ──────────────────────────────────────────────────────────
    print_summary(doc_path, vector_path)

    os.makedirs(os.path.dirname(vector_path) if os.path.dirname(vector_path) else ".", exist_ok=True)
    os.makedirs(vector_path, exist_ok=True)

    print("  ⏳  Starting ingestion...\n")
    t0 = time.time()

    try:
        ok = process_and_store_documents(doc_path, vector_path)
    except Exception as e:
        print(f"\n❌  Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0

    print()
    if ok:
        print(f"  ✅  Ingestion complete in {elapsed:.1f}s")
        print(f"  🔍  Vector store ready at: {vector_path}")
    else:
        print(f"  ⚠️   Ingestion ran but no documents were indexed.")
        print(f"       Check that your files are not empty or password-protected.")
    print()


if __name__ == "__main__":
    main()