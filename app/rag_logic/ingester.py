# app/rag_logic/ingester.py

import os
import shutil
import json
import pandas as pd
import time
import random
import concurrent.futures

from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from .llm_factory import get_llm, get_embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)

from langchain.docstore.document import Document
from pydantic import BaseModel, Field

from .custom_loaders import BetterPDFLoader, BetterPowerPointLoader, LoaderConfig
from .path_utils import norm_path
from .bm25_index import persist_bm25_index

import re
import hashlib



# ============================================================
# Metadata helpers
# ============================================================
def sanitize_metadata(doc: Document):
    """Limpia metadatos complejos para evitar errores en ChromaDB."""
    if not hasattr(doc, "metadata") or not doc.metadata:
        return
    clean_meta = {}
    for key, value in doc.metadata.items():
        if isinstance(value, list):
            clean_meta[key] = ", ".join([str(v) for v in value])
        elif isinstance(value, dict):
            clean_meta[key] = str(value)
        elif value is None:
            clean_meta[key] = ""
        else:
            clean_meta[key] = value
    doc.metadata = clean_meta


# ============================================================
# Loaders
# ============================================================
def load_documents_from_path(data_path: str) -> List[Document]:
    """Load every supported document type."""
    documents: List[Document] = []
    print(f"Scanning root directory: {data_path}")

    loader_cfg = LoaderConfig()

    for root, dirs, files in os.walk(data_path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            if file.startswith(".") or file.startswith("~$"):
                continue

            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, data_path)
            folder_structure = os.path.dirname(relative_path)

            try:
                ext = file.lower().split(".")[-1]
                loader = None
                new_docs: Optional[List[Document]] = None

                if ext == "pdf":
                    loader = BetterPDFLoader(file_path, loader_cfg=loader_cfg)
                elif ext in ["txt", "md", "html", "htm"]:
                    loader = TextLoader(file_path, encoding="utf-8")
                elif ext == "docx":
                    loader = Docx2txtLoader(file_path)
                elif ext == "csv":
                    loader = CSVLoader(file_path, encoding="utf-8")
                elif ext in ["pptx", "ppt"]:
                    loader = BetterPowerPointLoader(file_path, loader_cfg=loader_cfg)
                elif ext in ["xlsx", "xls"]:
                    try:
                        excel_sheets = pd.read_excel(file_path, sheet_name=None)
                        temp_docs = []
                        for sheet_name, df in excel_sheets.items():
                            if df is None or df.empty:
                                continue
                            table_text = df.to_csv(index=False)
                            temp_docs.append(Document(
                                page_content=f"Archivo Excel: {file}\nHoja: {sheet_name}\n\n{table_text}",
                                metadata={
                                    "source": file_path, "relative_path": relative_path,
                                    "folder_context": folder_structure, "filename": file,
                                    "sheet_name": sheet_name, "file_type": "excel",
                                },
                            ))
                        new_docs = temp_docs
                    except Exception as e:
                        print(f"⚠️ Error leyendo Excel {file_path}: {e}")
                        new_docs = []
                elif ext in ["png", "jpg", "jpeg", "gif", "tif", "tiff"]:
                    new_docs = [Document(
                        page_content=f"Imagen sin texto OCR. Archivo: {file}",
                        metadata={
                            "source": file_path, "relative_path": relative_path,
                            "folder_context": folder_structure, "filename": file,
                            "file_type": "image",
                        },
                    )]

                if loader is not None:
                    new_docs = loader.load()
                if not new_docs:
                    continue

                for doc in new_docs:
                    sanitize_metadata(doc)
                    doc.metadata.setdefault("relative_path", relative_path)
                    doc.metadata.setdefault("folder_context", folder_structure)
                    doc.metadata.setdefault("filename", file)
                    doc.metadata.setdefault("file_stem", os.path.splitext(file)[0])
                    doc.metadata.setdefault("source", file_path)

                documents.extend(new_docs)
            except Exception as e:
                print(f"⚠️ Error cargando {file}: {e}")

    return documents


# ============================================================
# Text cleaning
# ============================================================
_SPACED_LETTERS_RE = re.compile(r"(?:\b[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{1,2}\b[\s]+){4,}", re.UNICODE)

def _despace_text(s: str) -> str:
    """Convierte secuencias tipo 'U N A  P R O P U E S T A'."""
    if not s: return s
    def _fix(match: re.Match) -> str:
        chunk = match.group(0)
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{1,2}|\s+", chunk)
        out = []
        prev_was_space = False
        for t in tokens:
            if t.isspace() or (not t.strip()):
                if len(t) >= 2 and not prev_was_space:
                    out.append(" ")
                    prev_was_space = True
                continue
            out.append(t)
            prev_was_space = False
        return "".join(out).strip()
    return _SPACED_LETTERS_RE.sub(_fix, s)

def _fix_common_spanish_ocr_glue(s: str) -> str:
    """
    Split words the OCR glued together in Spanish text, and CamelCase transitions.
    e.g. 'dela' -> 'de la', 'Laeconomía' -> 'La economía'.
    """
    if not s: return s
    
    # 1. Split accidental CamelCase (lowercase followed by uppercase)
    # e.g. "economíaLa" -> "economía La"
    s = re.sub(r'([a-záéíóúüñ])([A-ZÁÉÍÓÚÜÑ])', r'\1 \2', s)
    
    # 2. Split common Spanish prepositions and articles that got glued together
    # Only when surrounded by spaces or boundaries, so words like 'adela' survive
    glue_patterns = [
        (r'\bdela\b', 'de la'),
        (r'\bdelos\b', 'de los'),
        (r'\bdelas\b', 'de las'),
        (r'\bala\b', 'a la'),
        (r'\balos\b', 'a los'),
        (r'\balas\b', 'a las'),
        (r'\benla\b', 'en la'),
        (r'\bconla\b', 'con la'),
        (r'\bporla\b', 'por la'),
        (r'\bparaque\b', 'para que'),
        (r'\besla\b', 'es la'),
    ]
    
    for pattern, replacement in glue_patterns:
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)
        
    return s

def _normalize_caps_runs(s: str) -> str:
    """Convierte ALL CAPS largos a Title Case, manteniendo siglas cortas."""
    if not s: return s
    def _caps_to_title(match: re.Match) -> str:
        text = match.group(0)
        words = text.split()
        if all(len(w) <= 4 for w in words): return text
        result = []
        for w in words:
            if len(w) <= 4 and w.isupper(): result.append(w)
            else: result.append(w.capitalize())
        return " ".join(result)
    return re.sub(r'[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s]{39,}', _caps_to_title, s)

def _collapse_whitespace(s: str) -> str:
    if not s: return s
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = re.sub(r' {2,}', ' ', s)
    return s.strip()


def _clean_text_for_embeddings(s: str) -> str:
    """Pipeline completo de limpieza."""
    if not s: return s
    s = _despace_text(s)
    s = _fix_common_spanish_ocr_glue(s) 
    s = _normalize_caps_runs(s)
    s = _collapse_whitespace(s)
    return s



def _normalize_for_hash(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.lower().strip())


# ============================================================
# Cross-page deduplication (removes repeated boilerplate blocks)
# ============================================================
def _dedup_repeated_blocks_across_pages(documents: List[Document], min_block_words: int = 30) -> List[Document]:
    """
    Detect text blocks (>= min_block_words) that repeat identically
    across multiple pages of the same file, and removes them from the pages
    donde son redundantes (mantiene solo la primera ocurrencia).
    
    This is common in decks where the same slide template repeats
    (e.g. a footer or a section header carried on every slide).
    """
    # Group by file
    docs_by_source = {}
    for doc in documents:
        meta = doc.metadata or {}
        src = meta.get("relative_path") or meta.get("filename") or "unknown"
        docs_by_source.setdefault(src, []).append(doc)

    total_blocks_removed = 0

    for src, src_docs in docs_by_source.items():
        if len(src_docs) < 2:
            continue

        # Extract blocks of N+ words from each page
        page_blocks = {}  # block_normalized -> list of (doc_index, original_block)
        for idx, doc in enumerate(src_docs):
            text = doc.page_content or ""
            # Split into paragraphs (blocks separated by a blank line)
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                para_clean = para.strip()
                if len(para_clean.split()) < min_block_words:
                    continue
                normalized = _normalize_for_hash(para_clean)
                page_blocks.setdefault(normalized, []).append((idx, para_clean))

        # Encontrar bloques repetidos
        for normalized, occurrences in page_blocks.items():
            if len(occurrences) <= 1:
                continue

            # Keep the first occurrence, drop the rest
            for dup_idx, dup_text in occurrences[1:]:
                doc = src_docs[dup_idx]
                original = doc.page_content or ""
                # Remove the duplicated block
                cleaned = original.replace(dup_text, "").strip()
                cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
                if cleaned != original:
                    doc.page_content = cleaned
                    total_blocks_removed += 1

    if total_blocks_removed > 0:
        print(f"   🔄 Removed {total_blocks_removed} text blocks repeated across pages")

    return documents


# ============================================================
# Contamination detection (e.g. wrong company name from template reuse)
# ============================================================
def _detect_contamination(documents: List[Document]) -> None:
    """
    Detect possible content contamination (e.g. another company's name
    inside a document that should belong to another). Reports only; changes nothing.
    """
    # Collect every company or brand name that appears frequently
    all_text = " ".join((d.page_content or "") for d in documents).upper()

    # Suspicious pattern: company names appearing a handful of times
    # against the main name, which appears many times
    company_pattern = re.compile(r'\b([A-ZÁÉÍÓÚÜÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÜÑ]{2,}){1,3})\s+\d{4}\b')
    matches = company_pattern.findall(all_text)

    if not matches:
        return

    from collections import Counter
    company_counts = Counter(m.strip() for m in matches)

    if len(company_counts) <= 1:
        return

    # The main company is the most frequent one
    main_company = company_counts.most_common(1)[0][0]
    for company, count in company_counts.items():
        if company != main_company and count <= 5:
            # Find which pages it appears on
            affected_pages = []
            for d in documents:
                if company in (d.page_content or "").upper():
                    pg = (d.metadata or {}).get("page_number") or "?"
                    affected_pages.append(pg)
            print(
                f"   ⚠️ CONTAMINATION DETECTED: '{company}' appears {count}x "
                f"in a '{main_company}' document (pages: {affected_pages}). "
                f"Likely template reuse."
            )


# ============================================================
# Context injection (prepends SOURCE + TITLE to the text)
# ============================================================
def inject_context_to_chunks(chunks: List[Document]) -> List[Document]:
    """Prepend minimal context to the text without polluting the embedding."""
    print("🧠 Adding context (metadata + SOURCE line) to each fragment...")

    for chunk in chunks:
        meta = chunk.metadata or {}
        path = meta.get("relative_path", "Documento desconocido")

        # Use the chunk's page range
        pages_in_chunk = meta.get("pages_in_chunk", None)
        if isinstance(pages_in_chunk, str) and pages_in_chunk:
            pages_in_chunk = [int(p.strip()) for p in pages_in_chunk.split(",") if p.strip().isdigit()]

        if pages_in_chunk and len(pages_in_chunk) > 1:
            where = f"{path} | pp. {pages_in_chunk[0]}-{pages_in_chunk[-1]}"
        elif pages_in_chunk and len(pages_in_chunk) == 1:
            where = f"{path} | p. {pages_in_chunk[0]}"
        else:
            page_number = meta.get("page_number")
            page = meta.get("page")
            if page_number is not None:
                where = f"{path} | p. {page_number}"
            elif page is not None:
                where = f"{path} | p. {int(page) + 1}"
            else:
                where = f"{path}"

        meta["context_where"] = where
        chunk.metadata = meta

        text = (chunk.page_content or "").lstrip()
        headline = (meta.get("semantic_headline") or "").strip()
        # The summary is injected alongside the headline so that it reaches the
        # embedding: a user question resembles a headline and a summary far more
        # than it resembles the body of a document, and that is what brings the
        # chunk vector closer to the question vector.
        summary = (meta.get("semantic_summary") or "").strip()
        cabecera = f"SOURCE: {where}\n"
        if headline:
            cabecera += f"TITLE: {headline}\n"
        if summary:
            cabecera += f"SUMMARY: {summary}\n"
        chunk.page_content = f"{cabecera}\n{text}"

    return chunks


# ============================================================
# PHASE 1: deterministic chunking over adjacent pages
# ============================================================
def _page_based_chunking(
    documents: List[Document],
    min_words: int = 150,
    target_words: int = 350,
    max_words: int = 700,
) -> List[Document]:
    """
    DETERMINISTIC page-based chunking.

    PRINCIPIOS:
    - Each text page is the smallest atomic unit
    - Consecutive pages are grouped until target_words is reached
    - Empty pages (is_empty_page) are absorbed into the nearest chunk
    - GUARANTEES 100% coverage of every page that has text
    - Does NOT depend on any LLM to decide the boundaries

    The result is chunks of ~200-500 words, each of which
    maps to 1-4 consecutive pages of the document.
    """
    # Group by file fuente
    docs_by_source = {}
    for doc in documents:
        meta = doc.metadata or {}
        src = (
            meta.get("relative_path")
            or meta.get("filename")
            or meta.get("source")
            or "unknown"
        )
        docs_by_source.setdefault(src, []).append(doc)

    all_chunks: List[Document] = []

    for src, src_docs in docs_by_source.items():
        # Sort by page
        src_docs.sort(key=lambda d: int(
            (d.metadata or {}).get("page_number")
            or (d.metadata or {}).get("page")
            or (d.metadata or {}).get("slide_number")
            or (d.metadata or {}).get("slide")
            or 0
        ))

        # Separate pages with real content from empty ones
        content_pages = []
        empty_page_numbers = []

        for d in src_docs:
            meta = d.metadata or {}
            is_empty = meta.get("is_empty_page") or meta.get("is_empty_slide") or False
            t = (d.page_content or "").strip()

            # Filter out empty-page placeholders
            if is_empty or t.startswith("[Page ") or t.startswith("[Slide "):
                pg = meta.get("page_number") or meta.get("slide_number") or "?"
                empty_page_numbers.append(pg)
                continue

            if len(t.split()) < 3:
                pg = meta.get("page_number") or meta.get("slide_number") or "?"
                empty_page_numbers.append(pg)
                continue

            content_pages.append(d)

        if empty_page_numbers:
            print(f"   📋 Pages without text in '{src}': {empty_page_numbers} ({len(empty_page_numbers)} of {len(src_docs)})")

        if not content_pages:
            print(f"   ⚠️ '{src}' has no pages with content. Skipping.")
            continue

        # GROUP consecutive pages into chunks
        chunks_for_file: List[Document] = []
        current_pages: List[Document] = []
        current_word_count = 0

        for page_doc in content_pages:
            page_words = len((page_doc.page_content or "").split())

            # A single page over max_words stands alone
            if page_words >= max_words:
                # Primero flush lo acumulado
                if current_pages:
                    chunks_for_file.append(_merge_pages_into_chunk(current_pages, src))
                    current_pages = []
                    current_word_count = 0
                # The large page goes on its own
                chunks_for_file.append(_merge_pages_into_chunk([page_doc], src))
                continue

            # Add this page to the current group, or start a new one?
            would_be = current_word_count + page_words

            if current_pages and would_be > max_words:
                # Flush: the current group is already big enough
                chunks_for_file.append(_merge_pages_into_chunk(current_pages, src))
                current_pages = [page_doc]
                current_word_count = page_words
            elif current_pages and would_be >= target_words:
                # Include this page and flush: the target is reached
                current_pages.append(page_doc)
                chunks_for_file.append(_merge_pages_into_chunk(current_pages, src))
                current_pages = []
                current_word_count = 0
            else:
                # Seguir acumulando
                current_pages.append(page_doc)
                current_word_count += page_words

        # Flush final
        if current_pages:
            # If the last group is very short and there are previous chunks, merge
            if current_word_count < min_words and chunks_for_file:
                last_chunk = chunks_for_file[-1]
                last_words = len((last_chunk.page_content or "").split())
                if last_words + current_word_count <= max_words:
                    # Merge with the previous chunk
                    chunks_for_file.pop()
                    all_pages_docs = []
                    # Recover the previous chunk's pages from metadata
                    # Simpler: rebuild by merging the texts
                    merged_text = (last_chunk.page_content or "").rstrip() + "\n\n" + \
                        "\n\n".join(f"[PAGE {(d.metadata or {}).get('page_number', '?')}]\n{d.page_content}"
                                    for d in current_pages)
                    last_meta = dict(last_chunk.metadata or {})
                    # Update the page list
                    prev_pages = last_meta.get("pages_in_chunk", "")
                    if isinstance(prev_pages, str):
                        prev_pages_list = [int(p.strip()) for p in prev_pages.split(",") if p.strip().isdigit()]
                    else:
                        prev_pages_list = list(prev_pages) if prev_pages else []
                    for d in current_pages:
                        pg = (d.metadata or {}).get("page_number") or (d.metadata or {}).get("slide_number")
                        if pg:
                            prev_pages_list.append(int(pg))
                    last_meta["pages_in_chunk"] = ", ".join(str(p) for p in sorted(set(prev_pages_list)))
                    last_meta["word_count"] = len(merged_text.split())
                    last_meta["char_count"] = len(merged_text)
                    chunks_for_file.append(Document(page_content=merged_text, metadata=last_meta))
                else:
                    chunks_for_file.append(_merge_pages_into_chunk(current_pages, src))
            else:
                chunks_for_file.append(_merge_pages_into_chunk(current_pages, src))

        # Statistics
        chunk_words = [len((c.page_content or "").split()) for c in chunks_for_file]
        total_pages_covered = set()
        for c in chunks_for_file:
            pgs = (c.metadata or {}).get("pages_in_chunk", "")
            if isinstance(pgs, str):
                for p in pgs.split(","):
                    p = p.strip()
                    if p.isdigit():
                        total_pages_covered.add(int(p))

        print(
            f"📄 '{src}': {len(content_pages)} pages with text → {len(chunks_for_file)} chunks "
            f"(words: min={min(chunk_words)}, max={max(chunk_words)}, avg={sum(chunk_words)/len(chunk_words):.0f}) "
            f"| pages covered: {len(total_pages_covered)}/{len(src_docs)}"
        )

        all_chunks.extend(chunks_for_file)

    return all_chunks


def _merge_pages_into_chunk(pages: List[Document], src: str) -> Document:
    """
    Merge a list of Documents (consecutive pages) into ONE chunk.
    Adds [PAGE X] markers for traceability.
    """
    parts = []
    page_numbers = []
    base_meta = dict(pages[0].metadata or {})
    # has_table only when some page REALLY carries table content
    has_real_table = False

    for d in pages:
        meta = d.metadata or {}
        pg = meta.get("page_number") or meta.get("slide_number") or "?"
        page_numbers.append(int(pg) if str(pg).isdigit() else 0)
        page_text = (d.page_content or '').strip()
        parts.append(f"[PAGE {pg}]\n{page_text}")
        # Only mark a table when the text actually carries the marker
        if "[TABLA]" in page_text:
            has_real_table = True

    merged_text = "\n\n".join(parts)

    pages_str = ", ".join(str(p) for p in sorted(set(page_numbers)) if p > 0)

    # Drop metadata inherited from the first page that does not apply to the chunk
    for stale_key in ["is_empty_page", "is_empty_slide", "text_chars",
                       "page", "page_number", "slide", "slide_number",
                       "ocr_used_page", "ocr_used_slide"]:
        base_meta.pop(stale_key, None)

    base_meta["pages_in_chunk"] = pages_str
    base_meta["chunking"] = "page_based_hybrid"
    base_meta["word_count"] = len(merged_text.split())
    base_meta["char_count"] = len(merged_text)
    base_meta["has_table"] = has_real_table
    base_meta["semantic_source"] = src

    # Stable chunk_id (hash of content + position)
    hash_input = f"{src}|{pages_str}|{merged_text[:200]}"
    base_meta["chunk_id"] = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:16]

    return Document(page_content=merged_text, metadata=base_meta)


# ============================================================
# PHASE 1.5: micro chunks (children) for retrieval precision
# ============================================================
def _generate_micro_chunks(
    macro_chunks: List[Document],
    target_tokens: int = 250,
    overlap_tokens: int = 60,
) -> List[Document]:
    """
    Generate "child" micro chunks from each macro chunk, for retrieval precision.

    ARQUITECTURA PARENT-CHILD:
    - Macro chunk (parent): ~250-500 words over 2-4 pages -> good for context and citations
    - Micro chunk (child): ~150-300 words with overlap -> good for precise retrieval
    - Cada child lleva parent_chunk_id para reconstruir contexto

    RETRIEVAL PATTERN:
      1. Search by micro chunk (high precision)
      2. Expandir a parent para contexto completo
      3. Citar con pages_in_chunk del parent
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n[PAGE ", "\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    all_micro: List[Document] = []
    parents_with_children = 0

    for parent in macro_chunks:
        parent_meta = parent.metadata or {}
        parent_id = parent_meta.get("chunk_id", "")
        parent_text = parent.page_content or ""

        # Take only the clean text; SOURCE/TITLE headers have not been added yet
        # The [PAGE X] markers are kept for traceability
        splits = splitter.split_text(parent_text)

        if len(splits) <= 1:
            # The parent is already small enough: no micro chunks needed
            continue

        parents_with_children += 1

        for micro_i, micro_text in enumerate(splits, start=1):
            micro_text = micro_text.strip()
            if not micro_text or len(micro_text.split()) < 20:
                continue

            # Extract the pages referenced by this micro chunk
            micro_pages = re.findall(r'\[PAGE\s+(\d+)\]', micro_text)
            micro_pages_str = ", ".join(sorted(set(micro_pages))) if micro_pages else parent_meta.get("pages_in_chunk", "")

            # Generar micro_chunk_id estable
            micro_hash_input = f"{parent_id}|{micro_i}|{micro_text[:100]}"
            micro_id = hashlib.sha1(micro_hash_input.encode("utf-8")).hexdigest()[:16]

            micro_meta = {
                "chunk_type": "micro",
                "parent_chunk_id": parent_id,
                "parent_pages": parent_meta.get("pages_in_chunk", ""),
                "chunk_id": micro_id,
                "micro_chunk_index": micro_i,
                "pages_in_chunk": micro_pages_str,
                "chunking": "micro_token_split",
                "semantic_source": parent_meta.get("semantic_source", ""),
                "source": parent_meta.get("source", ""),
                "source_file": parent_meta.get("source_file", ""),
                "relative_path": parent_meta.get("relative_path", ""),
                "filename": parent_meta.get("filename", ""),
                "file_stem": parent_meta.get("file_stem", ""),
                "file_type": parent_meta.get("file_type", ""),
                "folder_context": parent_meta.get("folder_context", ""),
                # Heredar headline/summary del parent
                "semantic_headline": parent_meta.get("semantic_headline", ""),
                "semantic_summary": parent_meta.get("semantic_summary", ""),
                "word_count": len(micro_text.split()),
                "char_count": len(micro_text),
            }

            all_micro.append(Document(page_content=micro_text, metadata=micro_meta))

        # Mark the parent as such
        parent_meta["chunk_type"] = "macro"
        parent_meta["has_children"] = True
        parent.metadata = parent_meta

    # Parents with no children (already small) are marked too
    for parent in macro_chunks:
        parent_meta = parent.metadata or {}
        if "chunk_type" not in parent_meta:
            parent_meta["chunk_type"] = "macro"
            parent_meta["has_children"] = False
            parent.metadata = parent_meta

    print(
        f"🔬 Micro chunks generated: {len(all_micro)} children from "
        f"{parents_with_children}/{len(macro_chunks)} parents"
    )

    if all_micro:
        micro_words = [len((m.page_content or "").split()) for m in all_micro]
        print(
            f"   • Words per micro chunk: min={min(micro_words)}, max={max(micro_words)}, "
            f"avg={sum(micro_words)/len(micro_words):.0f}"
        )

    return all_micro

# ============================================================
# PHASE 2: LLM enrichment (headline + summary only)
# ============================================================
class ChunkEnrichment(BaseModel):
    headline: str = Field(description="Short, descriptive title for the content (max 12 words)")
    summary: str = Field(description="A 2-4 sentence retrieval-oriented summary: topic, key figures, conclusion")


class ChunkEnrichmentList(BaseModel):
    enrichments: List[ChunkEnrichment] = Field(description="List of enrichments, one per chunk")


def _default_semantic_llm_model() -> str:
    return os.environ.get("SEMANTIC_CHUNKING_MODEL", "gpt-4o-mini")


def _enrichment_language() -> str:
    """
    Language for the generated headline and summary.

    Explicit on purpose. The instruction used to be "write in the same language
    as the chunk content", which delegated the choice to the model — and the
    model followed the language of the prompt rather than of the chunk. Since
    the headline and summary are prepended to the text BEFORE embedding, the
    wrong language pulls every chunk vector away from the questions it should
    match, with no error and nothing in the logs to point at it.

    It should match the language the questions are asked in, which is not
    necessarily the language of the corpus.
    """
    return (os.environ.get("ENRICHMENT_LANGUAGE") or "English").strip() or "English"


def _should_use_semantic_enrichment() -> bool:
    v = str(os.environ.get("SEMANTIC_CHUNKING", "1")).strip()
    return v not in ("0", "false", "False", "no", "NO")


INSPECT_SEMANTIC_CHUNKS = str(os.environ.get("DEBUG_SEMANTIC_CHUNKS", "0")).strip() in ("1", "true", "True", "yes", "YES")


def _invoke_with_retries(structured_llm, messages, max_retries: int = 3):
    last_err = None
    for i in range(max_retries):
        try:
            return structured_llm.invoke(messages)
        except Exception as e:
            last_err = e
            time.sleep(0.8 + random.random() * 0.8)
    raise last_err


def enrich_chunks_with_llm(
    chunks: List[Document],
    llm_model_name: Optional[str] = None,
    temperature: float = 0.0,
    batch_size: int = 8,
) -> List[Document]:
    """
    PHASE 2: enrich chunks with an LLM-generated headline and summary.
    
    THE LLM DOES NOT TOUCH THE TEXT — it only generates semantic metadata.
    That is fast, cheap, and cannot lose content.
    
    Sent in BATCHES of N chunks to minimise API calls.
    """
    if not chunks:
        return chunks

    model_name = llm_model_name or _default_semantic_llm_model()
    llm = get_llm(model_name, temperature)
    structured_llm = llm.with_structured_output(ChunkEnrichmentList)

    language = _enrichment_language()

    system_prompt = (
        "You are an expert in RAG (Retrieval Augmented Generation).\n"
        "Your task is to generate high-quality metadata for text chunks used in semantic search.\n\n"

        "For EACH chunk provided, generate:\n"
        "1. 'headline': a descriptive title of at most 12 words. It must be specific.\n"
        "   - GOOD: 'Parental leave entitlement by caregiver type and notice period'\n"
        "   - BAD: 'Leave information'\n\n"

        "2. 'summary': a 2-4 sentence retrieval-oriented summary. It must include:\n"
        "   - The main topic of the chunk\n"
        "   - Key quantitative data if present (figures, percentages, KPIs)\n"
        "   - The main conclusion or insight\n"
        "   - Temporal or organisational context when the chunk carries it\n\n"

        "RULES:\n"
        "- One headline + summary for EACH chunk, in the order provided\n"
        "- The number of enrichments must EXACTLY equal the number of chunks\n"
        "- [PAGE X] markers indicate the source — use them for context but do not include them in headline/summary\n"
        f"- Write the headline and the summary in {language}, whatever the language of the chunk\n"
    )

    workers = max(2, min((os.cpu_count() or 4), 8))
    if INSPECT_SEMANTIC_CHUNKS:
        workers = 1

    print(f"🏷️ LLM enrichment: {len(chunks)} chunks in batches of {batch_size} (workers={workers})")

    # Split into batches
    batches = []
    for i in range(0, len(chunks), batch_size):
        batches.append((i, chunks[i:i + batch_size]))

    enrichment_results = {}  # idx -> (headline, summary)

    def _process_batch(batch_start: int, batch_chunks: List[Document]) -> dict:
        """Process one batch of chunks and return their enrichments."""
        # Build the prompt from the chunk texts
        chunk_texts = []
        for j, ch in enumerate(batch_chunks):
            text = (ch.page_content or "").strip()
            # Truncate very long texts so the context does not overflow
            if len(text) > 3000:
                text = text[:3000] + "..."
            chunk_texts.append(f"--- CHUNK {j+1} ---\n{text}")

        user_prompt = (
            f"Genera headline y summary para cada uno de los {len(batch_chunks)} chunks siguientes.\n"
            f"Devuelve EXACTAMENTE {len(batch_chunks)} enrichments en el mismo orden.\n\n"
            + "\n\n".join(chunk_texts)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result: ChunkEnrichmentList = _invoke_with_retries(structured_llm, messages)

        batch_results = {}
        for j, enrichment in enumerate(result.enrichments):
            if j < len(batch_chunks):
                global_idx = batch_start + j
                batch_results[global_idx] = (
                    (enrichment.headline or "").strip(),
                    (enrichment.summary or "").strip(),
                )

        return batch_results

    # Run the batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_process_batch, b_start, b_chunks): b_start
            for b_start, b_chunks in batches
        }

        for future in concurrent.futures.as_completed(future_map):
            b_start = future_map[future]
            try:
                batch_results = future.result()
                enrichment_results.update(batch_results)
            except Exception as e:
                print(f"   ⚠️ Enrichment failed for the batch starting at {b_start}: {e}")
                # Fallback: generate a basic headline and summary
                batch_end = min(b_start + batch_size, len(chunks))
                for idx in range(b_start, batch_end):
                    text = (chunks[idx].page_content or "")[:200]
                    enrichment_results[idx] = (
                        text[:80].replace("\n", " ").strip(),
                        "",
                    )

    # Apply the enrichments to the chunks
    enriched_count = 0
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        if idx in enrichment_results:
            headline, summary = enrichment_results[idx]
            meta["semantic_headline"] = headline
            meta["semantic_summary"] = summary
            enriched_count += 1
        else:
            # Fallback: use the first words as the headline
            text = (chunk.page_content or "")[:100].replace("\n", " ").strip()
            meta["semantic_headline"] = text[:80]
            meta["semantic_summary"] = ""
        chunk.metadata = meta

    print(f"✅ Enrichment complete: {enriched_count}/{len(chunks)} chunks enriched")

    if INSPECT_SEMANTIC_CHUNKS:
        for i, ch in enumerate(chunks):
            m = ch.metadata or {}
            print(f"\n  🔹 Chunk {i+1}")
            print(f"  HEADLINE: {m.get('semantic_headline', '')}")
            print(f"  SUMMARY : {m.get('semantic_summary', '')}")
            print(f"  PAGES   : {m.get('pages_in_chunk', '')}")
            print(f"  TEXT    : {(ch.page_content or '')[:300]}...")

    return chunks


# ============================================================
# Postprocess: enrich metadata
# ============================================================
def _enrich_chunk_metadata(docs: List[Document]) -> List[Document]:
    for doc in docs:
        meta = doc.metadata or {}
        text = doc.page_content or ""
        meta["word_count"] = len(text.split())
        meta["char_count"] = len(text)

        # contains_metrics: presence of quantitative data (numbers with %, €, $, or KPIs)
        meta["contains_metrics"] = bool(re.search(
            r'\d+[.,]?\d*\s*%'           # porcentajes: 15%, 2.7%
            r'|\d+[.,]?\d*\s*[€$]'       # moneda: 331€, 52.000$
            r'|[€$]\s*\d+[.,]?\d*'        # moneda prefijo: €331, $52k
            r'|\d+[.,]?\d*\s*[kKmM]\b'    # abreviaciones: 331K, 1M
            r'|ROAS|ROI|CPC|CTR|CPA|CPM|CPL|CVR'  # KPIs de marketing
            r'|\d+\.\d{3}'                 # miles con punto: 6.019, 50.441
            , text
        ))

        # ✅ contains_list: bullets o enumeraciones reales
        meta["contains_list"] = bool(re.search(
            r'(?:^|\n)\s*[-•●▸▹►]\s+'    # bullets
            r'|(?:^|\n)\s*\d+[.)]\s+'     # numeradas: 1) o 1.
            r'|(?:^|\n)\s*[a-z][.)]\s+'   # letras: a) o a.
            , text
        ))

        doc.metadata = meta
    return docs


# ============================================================
# Save chunks (debug/eval)
# ============================================================
def _save_semantic_chunks(chunks: List[Document], base_dir: str):
    from pathlib import Path
    out_dir = Path(base_dir) / "semantic_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_file = {}
    for ch in chunks:
        meta = ch.metadata or {}
        src = meta.get("semantic_source") or meta.get("relative_path") or meta.get("filename") or "unknown"
        by_file.setdefault(src, []).append({
            "headline": meta.get("semantic_headline"),
            "summary": meta.get("semantic_summary"),
            "text": ch.page_content,
            "pages_in_chunk": meta.get("pages_in_chunk"),
            "word_count": len((ch.page_content or "").split()),
            "char_count": len(ch.page_content or ""),
        })

    for src, items in by_file.items():
        safe_name = src.replace("\\", "_").replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"source_file": src, "num_chunks": len(items), "chunks": items},
                      f, ensure_ascii=False, indent=2)


def _save_final_chunks(chunks: List[Document], base_dir: str):
    from pathlib import Path
    out_dir = Path(base_dir) / "final_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_file = {}
    for ch in chunks:
        meta = ch.metadata or {}
        src = meta.get("source_file") or meta.get("relative_path") or meta.get("filename") or "unknown"
        by_file.setdefault(src, []).append({
            "text": ch.page_content,
            "metadata": {k: v for k, v in meta.items()},
        })

    for src, items in by_file.items():
        safe_name = src.replace("\\", "_").replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"source_file": src, "num_chunks": len(items), "chunks": items},
                      f, ensure_ascii=False, indent=2)


# ============================================================
# Main pipeline incremental
# ============================================================
def process_and_store_documents(data_path: str, vector_store_path: str) -> bool:
    """
    HYBRID pipeline:
      PHASE 1: deterministic page-based chunking (100% coverage guaranteed)
      PHASE 2: LLM enrichment (headline + summary only; never loses text)
      PHASE 3: context injection + storage in ChromaDB
    """

    def _safe_norm(p: str) -> str:
        return (p or "").replace("\\", "/").strip()

    def _manifest_path(vs_path: str) -> str:
        return os.path.join(vs_path, "_manifest.json")

    def _load_manifest(vs_path: str) -> dict:
        mp = _manifest_path(vs_path)
        if os.path.exists(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"version": 1, "files": {}}
        return {"version": 1, "files": {}}

    def _save_manifest(vs_path: str, manifest: dict) -> None:
        os.makedirs(vs_path, exist_ok=True)
        with open(_manifest_path(vs_path), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _file_sig(abs_path: str) -> dict:
        st = os.stat(abs_path)
        return {"mtime": int(st.st_mtime), "size": int(st.st_size)}

    def _get_rel(abs_path: str, base_dir: str) -> str:
        try:
            rel = os.path.relpath(abs_path, base_dir)
        except Exception:
            rel = os.path.basename(abs_path)
        return _safe_norm(rel)

    def _chunk_source_rel(chunk: Document, base_dir: str) -> str:
        meta = chunk.metadata or {}
        src = meta.get("source") or meta.get("filepath") or meta.get("path") or ""
        if src and os.path.isabs(src) and os.path.exists(src):
            return _get_rel(src, base_dir)
        rel = meta.get("relative_path") or meta.get("filename") or ""
        return _safe_norm(rel) or "unknown"

    def _delete_by_source_file(vs: Chroma, source_file: str) -> None:
        source_file = _safe_norm(source_file)
        if not source_file:
            return
        try:
            col = getattr(vs, "_collection", None)
            if col is not None:
                col.delete(where={"source_file": source_file})
                return
        except Exception:
            pass
        try:
            d = vs.get(include=["metadatas"])
            ids = []
            for _id, md in zip(d.get("ids", []), d.get("metadatas", [])):
                if (md or {}).get("source_file") == source_file:
                    ids.append(_id)
            if ids:
                vs.delete(ids=ids)
        except Exception:
            pass

    def _backfill_relative_path_norm(vs: Chroma) -> int:
        """Backfill relative_path_norm/department on already-indexed chunks, without re-embedding."""
        try:
            data = vs.get(include=["metadatas"])
        except Exception:
            return 0
        ids = data.get("ids", []) or []
        metas = data.get("metadatas", []) or []
        update_ids, update_metas = [], []
        for _id, m in zip(ids, metas):
            m = m or {}
            rel = m.get("relative_path") or m.get("source_file") or m.get("filename") or ""
            changed = False
            if not m.get("relative_path_norm") and rel:
                m["relative_path_norm"] = norm_path(rel)
                changed = True
            if not m.get("department") and rel:
                m["department"] = (_safe_norm(rel).split("/")[0] if "/" in _safe_norm(rel) else "general").lower()
                changed = True
            if changed:
                update_ids.append(_id)
                update_metas.append(m)
        if update_ids:
            col = getattr(vs, "_collection", None)
            if col is not None:
                col.update(ids=update_ids, metadatas=update_metas)
                print(f"🩹 Backfill relative_path_norm/department: {len(update_ids)} chunk(s) actualizados.")
        return len(update_ids)

    try:
        print(f"🚀 Ingesta Avanzada (INCREMENTAL) para: {data_path}")

        if not os.path.exists(data_path):
            print(f"❌ data_path no existe: {data_path}")
            return False

        documents = load_documents_from_path(data_path)
        if not documents:
            print("⚠️ No valid documents found.")
            return False

        os.makedirs(vector_store_path, exist_ok=True)
        manifest = _load_manifest(vector_store_path)
        old_files = set((manifest.get("files") or {}).keys())

        current_files = {}
        for d in documents:
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("filepath") or meta.get("path") or ""
            if src and os.path.isabs(src) and os.path.exists(src):
                rel = _get_rel(src, data_path)
                current_files[rel] = src

        if not current_files:
            for d in documents:
                meta = d.metadata or {}
                rel = _safe_norm(meta.get("relative_path") or meta.get("filename") or "")
                if rel:
                    abs_guess = os.path.join(data_path, rel)
                    if os.path.exists(abs_guess):
                        current_files[rel] = abs_guess

        deleted_files = sorted(list(old_files - set(current_files.keys())))
        changed_files = []
        for rel, abs_path in current_files.items():
            sig = _file_sig(abs_path)
            prev = (manifest.get("files") or {}).get(rel)
            if (prev or {}).get("mtime") != sig["mtime"] or (prev or {}).get("size") != sig["size"]:
                changed_files.append(rel)

        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings,
        )

        # Backfill for chunks indexed before relative_path_norm existed, which the
        # native metadata prefilter needs. Updates metadata only; nothing is re-embedded.
        _backfill_relative_path_norm(vector_store)

        for rel in deleted_files:
            print(f"🗑️ Removing from the index: {rel}")
            _delete_by_source_file(vector_store, rel)
            (manifest.get("files") or {}).pop(rel, None)

        if not changed_files and not deleted_files:
            print("✅ No changes detected. The index is already up to date.")
            if not os.path.exists(os.path.join(vector_store_path, "_bm25_index.pkl")):
                print("🔎 Backfill: building the persistent BM25 index (first run)...")
                persist_bm25_index(vector_store, vector_store_path)
            return True

        changed_set = set(changed_files)

        def _doc_rel(d: Document) -> str:
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("filepath") or meta.get("path") or ""
            if src and os.path.isabs(src) and os.path.exists(src):
                return _get_rel(src, data_path)
            return _safe_norm(meta.get("relative_path") or meta.get("filename") or "")

        docs_to_index = [d for d in documents if _doc_rel(d) in changed_set]

        # Text pre-cleaning
        for d in docs_to_index:
            d.page_content = _clean_text_for_embeddings(d.page_content or "")

        print(f"🧾 Cambios detectados: {len(changed_files)} archivo(s) a re-indexar.")
        if not docs_to_index:
            print("⚠️ No documents found to re-index.")
            return False

        for rel in changed_files:
            print(f"♻️ Re-index: limpiando chunks previos de {rel}")
            _delete_by_source_file(vector_store, rel)

        # Detect contamination (reports only, changes nothing)
        _detect_contamination(docs_to_index)

        # Remove blocks repeated across pages (e.g. duplicated slide templates)
        docs_to_index = _dedup_repeated_blocks_across_pages(docs_to_index, min_block_words=25)

        # ================================================================
        # PHASE 1: deterministic page-based chunking
        # ================================================================
        print("\n📐 PHASE 1: deterministic page-based chunking...")
        page_chunks = _page_based_chunking(
            docs_to_index,
            min_words=100,
            target_words=250,
            max_words=500,
        )

        if not page_chunks:
            print("⚠️ No chunks were generated. Aborting.")
            return False

        # ================================================================
        # PHASE 1.5: micro chunks (children) for precise Q&A
        # ================================================================
        print("\n🔬 PHASE 1.5: generating micro chunks (children)...")
        micro_chunks = _generate_micro_chunks(
            page_chunks,
            target_tokens=250,
            overlap_tokens=60,
        )

        # ================================================================
        # PHASE 2: LLM enrichment (headline + summary)
        # ================================================================
        # Enrich the macro chunks (parents) with the LLM
        if _should_use_semantic_enrichment():
            print("\n🏷️ PHASE 2: LLM enrichment (headline + summary)...")
            try:
                page_chunks = enrich_chunks_with_llm(page_chunks, batch_size=8)
                # Propagate the parent's headline/summary to its micro chunks
                parent_enrichments = {}
                for ch in page_chunks:
                    pid = (ch.metadata or {}).get("chunk_id", "")
                    if pid:
                        parent_enrichments[pid] = {
                            "semantic_headline": (ch.metadata or {}).get("semantic_headline", ""),
                            "semantic_summary": (ch.metadata or {}).get("semantic_summary", ""),
                        }
                for micro in micro_chunks:
                    pid = (micro.metadata or {}).get("parent_chunk_id", "")
                    if pid in parent_enrichments:
                        micro.metadata["semantic_headline"] = parent_enrichments[pid]["semantic_headline"]
                        micro.metadata["semantic_summary"] = parent_enrichments[pid]["semantic_summary"]
            except Exception as e:
                print(f"   ⚠️ LLM enrichment failed ({e}). Chunks stored without headline/summary.")

        # Save the semantic chunks for evaluation
        _save_semantic_chunks(page_chunks, base_dir=vector_store_path)

        # Enrich metadata (metrics, lists, ...) at BOTH levels
        page_chunks = _enrich_chunk_metadata(page_chunks)
        micro_chunks = _enrich_chunk_metadata(micro_chunks)

        # ================================================================
        # PHASE 3: context injection + storage
        # ================================================================
        print("\n💾 PHASE 3: context injection + storage...")

        # Context injection for both levels
        macro_final = inject_context_to_chunks(page_chunks)
        micro_final = inject_context_to_chunks(micro_chunks)

        # Combine macro + micro: everything is indexed in ChromaDB
        final_chunks = macro_final + micro_final

        if not final_chunks:
            return False

        # Guardar chunks finales
        if str(os.environ.get("SAVE_FINAL_CHUNKS_JSON", "1")).strip() in ("1", "true", "True", "yes", "YES"):
            _save_final_chunks(final_chunks, base_dir=vector_store_path)

        # Normalise metadata for ChromaDB
        for ch in final_chunks:
            meta = ch.metadata or {}
            rel = _chunk_source_rel(ch, data_path)
            meta["source_file"] = rel
            meta["relative_path"] = rel
            meta["relative_path_norm"] = norm_path(rel)  # clave estable para prefiltro nativo en Chroma
            # department = the FIRST folder segment. os.path.dirname returns the
            # whole path, so with subfolders it would give "compensation/2026" instead
            # of "compensation", and the access filter would stop matching.
            meta["department"] = (_safe_norm(rel).split("/")[0] if "/" in _safe_norm(rel) else "general").lower()
            # ChromaDB does not support lists — these are already strings from _merge_pages_into_chunk
            ch.metadata = meta

        # Statistics finales
        macro_count = sum(1 for c in final_chunks if (c.metadata or {}).get("chunk_type") == "macro")
        micro_count = sum(1 for c in final_chunks if (c.metadata or {}).get("chunk_type") == "micro")
        word_counts = [len((c.page_content or "").split()) for c in final_chunks]
        macro_words = [len((c.page_content or "").split()) for c in final_chunks if (c.metadata or {}).get("chunk_type") == "macro"]
        micro_words = [len((c.page_content or "").split()) for c in final_chunks if (c.metadata or {}).get("chunk_type") == "micro"]

        print(f"\n📊 FINAL STATISTICS:")
        print(f"   • Input docs: {len(docs_to_index)} pages")
        print(f"   • Total chunks indexed: {len(final_chunks)}")
        print(f"   • Macro chunks (parents): {macro_count}")
        if macro_words:
            print(f"     Words: min={min(macro_words)}, max={max(macro_words)}, avg={sum(macro_words)/len(macro_words):.0f}")
        print(f"   • Micro chunks (children): {micro_count}")
        if micro_words:
            print(f"     Words: min={min(micro_words)}, max={max(micro_words)}, avg={sum(micro_words)/len(micro_words):.0f}")
        with_headline = sum(1 for c in final_chunks if (c.metadata or {}).get("semantic_headline"))
        print(f"   • With headline: {with_headline}/{len(final_chunks)}")

        batch_size = 150
        total = len(final_chunks)
        for i in range(0, total, batch_size):
            batch = final_chunks[i: i + batch_size]
            vector_store.add_documents(batch)
            print(f"💾 Saving batch {i//batch_size + 1}. ({min(i + batch_size, total)}/{total})")

        files_dict = manifest.get("files") or {}
        for rel in changed_files:
            abs_path = current_files.get(rel)
            if abs_path and os.path.exists(abs_path):
                files_dict[rel] = _file_sig(abs_path)
        manifest["files"] = files_dict
        _save_manifest(vector_store_path, manifest)

        print("\n🔎 Rebuilding the persistent BM25 index...")
        if persist_bm25_index(vector_store, vector_store_path):
            print("✅ BM25 index persisted to disk.")
        else:
            print("⚠️ Could not persist the BM25 index (empty vector store?).")

        print(f"\n✅ Incremental ingestion finished at: {vector_store_path}")
        return True

    except Exception as e:
        print(f"❌ CRITICAL error during incremental ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False