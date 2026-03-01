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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)

from langchain.docstore.document import Document
from pydantic import BaseModel, Field

from .custom_loaders import BetterPDFLoader, BetterPowerPointLoader, OcrConfig

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
    """Carga documentos soportados con OCR selectivo."""
    documents: List[Document] = []
    print(f"Escaneando directorio raíz: {data_path}")

    ocr_cfg = OcrConfig(
        enabled=str(os.environ.get("OCR_ENABLED", "1")).strip() in ("1", "true", "True", "yes", "YES")
    )

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
                    loader = BetterPDFLoader(file_path, ocr_cfg=ocr_cfg)
                elif ext in ["txt", "md", "html", "htm"]:
                    loader = TextLoader(file_path, encoding="utf-8")
                elif ext == "docx":
                    loader = Docx2txtLoader(file_path)
                elif ext == "csv":
                    loader = CSVLoader(file_path, encoding="utf-8")
                elif ext in ["pptx", "ppt"]:
                    loader = BetterPowerPointLoader(file_path, ocr_cfg=ocr_cfg)
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
    Separa palabras pegadas típicas del OCR en español y transiciones CamelCase.
    Ej: 'dela' -> 'de la', 'Laeconomía' -> 'La economía'.
    """
    if not s: return s
    
    # 1. Separar CamelCase accidental (minúscula seguida de Mayúscula)
    # Ej: "economíaLa" -> "economía La"
    s = re.sub(r'([a-záéíóúüñ])([A-ZÁÉÍÓÚÜÑ])', r'\1 \2', s)
    
    # 2. Separar preposiciones/artículos pegados comunes en español
    # Solo si están rodeados de espacios o bordes (para no romper palabras como 'adela')
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
    s = _fix_common_spanish_ocr_glue(s) # ✅ Nueva función insertada
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
    Detecta bloques de texto (≥min_block_words) que se repiten idénticos
    en múltiples páginas del mismo archivo y los elimina de las páginas
    donde son redundantes (mantiene solo la primera ocurrencia).
    
    Esto es común en presentaciones donde se repite la misma slide-template
    (ej: 'DIMENSIÓN PEAK SALES... Buscadores Afiliación Programática...')
    """
    # Agrupar por archivo
    docs_by_source = {}
    for doc in documents:
        meta = doc.metadata or {}
        src = meta.get("relative_path") or meta.get("filename") or "unknown"
        docs_by_source.setdefault(src, []).append(doc)

    total_blocks_removed = 0

    for src, src_docs in docs_by_source.items():
        if len(src_docs) < 2:
            continue

        # Extraer bloques de N+ palabras de cada página
        page_blocks = {}  # block_normalized -> list of (doc_index, original_block)
        for idx, doc in enumerate(src_docs):
            text = doc.page_content or ""
            # Dividir en párrafos (bloques separados por doble salto de línea)
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

            # Mantener primera ocurrencia, eliminar el resto
            for dup_idx, dup_text in occurrences[1:]:
                doc = src_docs[dup_idx]
                original = doc.page_content or ""
                # Eliminar el bloque duplicado
                cleaned = original.replace(dup_text, "").strip()
                cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
                if cleaned != original:
                    doc.page_content = cleaned
                    total_blocks_removed += 1

    if total_blocks_removed > 0:
        print(f"   🔄 Eliminados {total_blocks_removed} bloques de texto repetidos entre páginas")

    return documents


# ============================================================
# Contamination detection (e.g. wrong company name from template reuse)
# ============================================================
def _detect_contamination(documents: List[Document]) -> None:
    """
    Detecta posibles contaminaciones de contenido (ej: nombre de otra empresa
    en un documento que debería ser de otra). Solo reporta, no modifica.
    """
    # Recoger todos los nombres de empresa/marca que aparecen frecuentemente
    all_text = " ".join((d.page_content or "") for d in documents).upper()

    # Patrones sospechosos: nombres de empresa que aparecen pocas veces
    # vs el nombre principal que aparece muchas veces
    company_pattern = re.compile(r'\b([A-ZÁÉÍÓÚÜÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÜÑ]{2,}){1,3})\s+\d{4}\b')
    matches = company_pattern.findall(all_text)

    if not matches:
        return

    from collections import Counter
    company_counts = Counter(m.strip() for m in matches)

    if len(company_counts) <= 1:
        return

    # La empresa principal es la más frecuente
    main_company = company_counts.most_common(1)[0][0]
    for company, count in company_counts.items():
        if company != main_company and count <= 5:
            # Buscar en qué páginas aparece
            affected_pages = []
            for d in documents:
                if company in (d.page_content or "").upper():
                    pg = (d.metadata or {}).get("page_number") or "?"
                    affected_pages.append(pg)
            print(
                f"   ⚠️ CONTAMINACIÓN DETECTADA: '{company}' aparece {count}x "
                f"en documento de '{main_company}' (páginas: {affected_pages}). "
                f"Probable reutilización de plantilla."
            )


# ============================================================
# Context injection (añade SOURCE + TITLE al texto)
# ============================================================
def inject_context_to_chunks(chunks: List[Document]) -> List[Document]:
    """Añade contexto mínimo al texto sin contaminar embeddings."""
    print("🧠 Añadiendo contexto (metadata + línea SOURCE) a cada fragmento...")

    for chunk in chunks:
        meta = chunk.metadata or {}
        path = meta.get("relative_path", "Documento desconocido")

        # ✅ Usar rango de páginas del chunk
        pages_in_chunk = meta.get("pages_in_chunk", None)
        if isinstance(pages_in_chunk, str) and pages_in_chunk:
            pages_in_chunk = [int(p.strip()) for p in pages_in_chunk.split(",") if p.strip().isdigit()]

        if pages_in_chunk and len(pages_in_chunk) > 1:
            where = f"{path} | págs. {pages_in_chunk[0]}-{pages_in_chunk[-1]}"
        elif pages_in_chunk and len(pages_in_chunk) == 1:
            where = f"{path} | pág. {pages_in_chunk[0]}"
        else:
            page_number = meta.get("page_number")
            page = meta.get("page")
            if page_number is not None:
                where = f"{path} | pág. {page_number}"
            elif page is not None:
                where = f"{path} | pág. {int(page) + 1}"
            else:
                where = f"{path}"

        meta["context_where"] = where
        chunk.metadata = meta

        text = (chunk.page_content or "").lstrip()
        headline = (meta.get("semantic_headline") or "").strip()
        headline_line = f"TITLE: {headline}\n" if headline else ""
        chunk.page_content = f"SOURCE: {where}\n{headline_line}\n{text}"

    return chunks


# ============================================================
# FASE 1: Chunking determinista por páginas adyacentes
# ============================================================
def _page_based_chunking(
    documents: List[Document],
    min_words: int = 150,
    target_words: int = 350,
    max_words: int = 700,
) -> List[Document]:
    """
    Chunking DETERMINISTA basado en páginas.

    PRINCIPIOS:
    - Cada página de texto es la unidad atómica mínima
    - Se agrupan páginas consecutivas hasta alcanzar target_words
    - Las páginas vacías (is_empty_page) se absorben en el chunk más cercano
    - GARANTIZA 100% de cobertura de todas las páginas con texto
    - NO depende de ningún LLM para decidir los límites

    El resultado son chunks de ~200-500 palabras donde cada uno
    corresponde a 1-4 páginas consecutivas del documento.
    """
    # Agrupar por archivo fuente
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
        # Ordenar por página
        src_docs.sort(key=lambda d: int(
            (d.metadata or {}).get("page_number")
            or (d.metadata or {}).get("page")
            or (d.metadata or {}).get("slide_number")
            or (d.metadata or {}).get("slide")
            or 0
        ))

        # Separar páginas con contenido real vs vacías
        content_pages = []
        empty_page_numbers = []

        for d in src_docs:
            meta = d.metadata or {}
            is_empty = meta.get("is_empty_page") or meta.get("is_empty_slide") or False
            t = (d.page_content or "").strip()

            # Filtrar placeholders de páginas vacías
            if is_empty or t.startswith("[Página ") or t.startswith("[Slide "):
                pg = meta.get("page_number") or meta.get("slide_number") or "?"
                empty_page_numbers.append(pg)
                continue

            if len(t.split()) < 3:
                pg = meta.get("page_number") or meta.get("slide_number") or "?"
                empty_page_numbers.append(pg)
                continue

            content_pages.append(d)

        if empty_page_numbers:
            print(f"   📋 Páginas sin texto en '{src}': {empty_page_numbers} ({len(empty_page_numbers)} de {len(src_docs)})")

        if not content_pages:
            print(f"   ⚠️ '{src}' no tiene páginas con contenido. Saltando.")
            continue

        # ✅ AGRUPAR páginas consecutivas en chunks
        chunks_for_file: List[Document] = []
        current_pages: List[Document] = []
        current_word_count = 0

        for page_doc in content_pages:
            page_words = len((page_doc.page_content or "").split())

            # Si una sola página ya excede max_words, va sola
            if page_words >= max_words:
                # Primero flush lo acumulado
                if current_pages:
                    chunks_for_file.append(_merge_pages_into_chunk(current_pages, src))
                    current_pages = []
                    current_word_count = 0
                # La página grande va sola
                chunks_for_file.append(_merge_pages_into_chunk([page_doc], src))
                continue

            # ¿Añadir esta página al grupo actual o empezar nuevo?
            would_be = current_word_count + page_words

            if current_pages and would_be > max_words:
                # Flush: el grupo actual ya es suficiente
                chunks_for_file.append(_merge_pages_into_chunk(current_pages, src))
                current_pages = [page_doc]
                current_word_count = page_words
            elif current_pages and would_be >= target_words:
                # Incluir esta página y flush (llegamos al target)
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
            # Si el último grupo es muy corto y hay chunks anteriores, fusionar
            if current_word_count < min_words and chunks_for_file:
                last_chunk = chunks_for_file[-1]
                last_words = len((last_chunk.page_content or "").split())
                if last_words + current_word_count <= max_words:
                    # Fusionar con el chunk anterior
                    chunks_for_file.pop()
                    all_pages_docs = []
                    # Recuperar las páginas del chunk anterior desde metadata
                    # Más simple: reconstruir fusionando textos
                    merged_text = (last_chunk.page_content or "").rstrip() + "\n\n" + \
                        "\n\n".join(f"[PÁGINA {(d.metadata or {}).get('page_number', '?')}]\n{d.page_content}"
                                    for d in current_pages)
                    last_meta = dict(last_chunk.metadata or {})
                    # Actualizar páginas
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

        # Estadísticas
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
            f"📄 '{src}': {len(content_pages)} páginas con texto → {len(chunks_for_file)} chunks "
            f"(palabras: min={min(chunk_words)}, max={max(chunk_words)}, avg={sum(chunk_words)/len(chunk_words):.0f}) "
            f"| páginas cubiertas: {len(total_pages_covered)}/{len(src_docs)}"
        )

        all_chunks.extend(chunks_for_file)

    return all_chunks


def _merge_pages_into_chunk(pages: List[Document], src: str) -> Document:
    """
    Fusiona una lista de Documents (páginas consecutivas) en UN chunk.
    Añade marcadores [PÁGINA X] para trazabilidad.
    """
    parts = []
    page_numbers = []
    base_meta = dict(pages[0].metadata or {})
    # ✅ has_table solo si alguna página REALMENTE tiene contenido [TABLA]
    has_real_table = False

    for d in pages:
        meta = d.metadata or {}
        pg = meta.get("page_number") or meta.get("slide_number") or "?"
        page_numbers.append(int(pg) if str(pg).isdigit() else 0)
        page_text = (d.page_content or '').strip()
        parts.append(f"[PÁGINA {pg}]\n{page_text}")
        # ✅ Solo marcar tabla si el texto realmente contiene marcador [TABLA]
        if "[TABLA]" in page_text:
            has_real_table = True

    merged_text = "\n\n".join(parts)

    pages_str = ", ".join(str(p) for p in sorted(set(page_numbers)) if p > 0)

    # ✅ Limpiar metadata heredada de la primera página que no aplica al chunk
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

    # ✅ Generar chunk_id estable (hash del contenido + posición)
    hash_input = f"{src}|{pages_str}|{merged_text[:200]}"
    base_meta["chunk_id"] = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:16]

    return Document(page_content=merged_text, metadata=base_meta)


# ============================================================
# FASE 1.5: Micro-chunks (child) para precisión en Q&A
# ============================================================
def _generate_micro_chunks(
    macro_chunks: List[Document],
    target_tokens: int = 250,
    overlap_tokens: int = 60,
) -> List[Document]:
    """
    Genera micro-chunks "hijo" de cada macro-chunk para Q&A preciso.

    ARQUITECTURA PARENT-CHILD:
    - Macro-chunk (parent): ~250-500 palabras, agrupa 2-4 páginas → bueno para contexto/citas
    - Micro-chunk (child): ~150-300 palabras con overlap → bueno para retrieval preciso
    - Cada child lleva parent_chunk_id para reconstruir contexto

    RETRIEVAL PATTERN:
      1. Buscar por micro-chunk (alta precisión)
      2. Expandir a parent para contexto completo
      3. Citar con pages_in_chunk del parent
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n[PÁGINA ", "\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )

    all_micro: List[Document] = []
    parents_with_children = 0

    for parent in macro_chunks:
        parent_meta = parent.metadata or {}
        parent_id = parent_meta.get("chunk_id", "")
        parent_text = parent.page_content or ""

        # Extraer solo el texto limpio (sin SOURCE/TITLE headers que aún no se han añadido)
        # Los marcadores [PÁGINA X] se mantienen para trazabilidad
        splits = splitter.split_text(parent_text)

        if len(splits) <= 1:
            # Parent ya es suficientemente pequeño — no generar micro-chunks
            continue

        parents_with_children += 1

        for micro_i, micro_text in enumerate(splits, start=1):
            micro_text = micro_text.strip()
            if not micro_text or len(micro_text.split()) < 20:
                continue

            # Extraer páginas referenciadas en este micro-chunk
            micro_pages = re.findall(r'\[PÁGINA\s+(\d+)\]', micro_text)
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

        # Marcar el parent como tal
        parent_meta["chunk_type"] = "macro"
        parent_meta["has_children"] = True
        parent.metadata = parent_meta

    # Parents sin children (ya eran pequeños) también se marcan
    for parent in macro_chunks:
        parent_meta = parent.metadata or {}
        if "chunk_type" not in parent_meta:
            parent_meta["chunk_type"] = "macro"
            parent_meta["has_children"] = False
            parent.metadata = parent_meta

    print(
        f"🔬 Micro-chunks generados: {len(all_micro)} children de "
        f"{parents_with_children}/{len(macro_chunks)} parents"
    )

    if all_micro:
        micro_words = [len((m.page_content or "").split()) for m in all_micro]
        print(
            f"   • Palabras/micro-chunk: min={min(micro_words)}, max={max(micro_words)}, "
            f"avg={sum(micro_words)/len(micro_words):.0f}"
        )

    return all_micro

# ============================================================
# FASE 2: Enriquecimiento LLM (solo headline + summary)
# ============================================================
class ChunkEnrichment(BaseModel):
    headline: str = Field(description="Título corto y descriptivo del contenido (máx 12 palabras)")
    summary: str = Field(description="Resumen de 2-4 frases orientado a recuperación RAG: tema, datos clave, conclusión")


class ChunkEnrichmentList(BaseModel):
    enrichments: List[ChunkEnrichment] = Field(description="Lista de enriquecimientos, uno por chunk")


def _default_semantic_llm_model() -> str:
    return os.environ.get("SEMANTIC_CHUNKING_MODEL", "gpt-4o-mini")


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
    FASE 2: Enriquece chunks con headline y summary generados por LLM.
    
    EL LLM NO TOCA EL TEXTO — solo genera metadata semántica.
    Esto es rápido, barato, y no puede perder contenido.
    
    Se envían en BATCHES de N chunks para minimizar llamadas API.
    """
    if not chunks:
        return chunks

    model_name = llm_model_name or _default_semantic_llm_model()
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    structured_llm = llm.with_structured_output(ChunkEnrichmentList)

    system_prompt = (
        "Eres un experto en RAG (Retrieval Augmented Generation).\n"
        "Tu tarea es generar metadatos de calidad para chunks de texto que se usarán en búsqueda semántica.\n\n"
        
        "Para CADA chunk proporcionado, genera:\n"
        "1. 'headline': Título descriptivo de máximo 12 palabras. Debe ser específico.\n"
        "   - BUENO: 'Resultados PMAX por categoría H&A, HE e IT 2025'\n"
        "   - MALO: 'Datos de resultados'\n\n"
        
        "2. 'summary': Resumen de 2-4 frases orientado a recuperación. Debe incluir:\n"
        "   - Tema principal del chunk\n"
        "   - Datos cuantitativos clave si los hay (cifras, porcentajes, KPIs)\n"
        "   - Conclusión o insight principal\n"
        "   - Contexto temporal/empresarial (ej: 'LG Electronics 2025')\n\n"
        
        "REGLAS:\n"
        "- Un headline + summary por CADA chunk en el orden proporcionado\n"
        "- El número de enrichments debe ser EXACTAMENTE igual al número de chunks\n"
        "- Los marcadores [PÁGINA X] indican la fuente — úsalos para contexto pero no los incluyas en headline/summary\n"
        "- Escribe en el mismo idioma que el contenido del chunk\n"
    )

    workers = max(2, min((os.cpu_count() or 4), 8))
    if INSPECT_SEMANTIC_CHUNKS:
        workers = 1

    print(f"🏷️ Enriquecimiento LLM: {len(chunks)} chunks en batches de {batch_size} (workers={workers})")

    # Dividir en batches
    batches = []
    for i in range(0, len(chunks), batch_size):
        batches.append((i, chunks[i:i + batch_size]))

    enrichment_results = {}  # idx -> (headline, summary)

    def _process_batch(batch_start: int, batch_chunks: List[Document]) -> dict:
        """Procesa un batch de chunks y devuelve sus enriquecimientos."""
        # Construir prompt con los textos de los chunks
        chunk_texts = []
        for j, ch in enumerate(batch_chunks):
            text = (ch.page_content or "").strip()
            # Truncar textos muy largos para no reventar contexto
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

    # Ejecutar batches en paralelo
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
                print(f"   ⚠️ Enriquecimiento falló para batch starting at {b_start}: {e}")
                # Fallback: generar headline/summary básicos
                batch_end = min(b_start + batch_size, len(chunks))
                for idx in range(b_start, batch_end):
                    text = (chunks[idx].page_content or "")[:200]
                    enrichment_results[idx] = (
                        text[:80].replace("\n", " ").strip(),
                        "",
                    )

    # Aplicar enriquecimientos a los chunks
    enriched_count = 0
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        if idx in enrichment_results:
            headline, summary = enrichment_results[idx]
            meta["semantic_headline"] = headline
            meta["semantic_summary"] = summary
            enriched_count += 1
        else:
            # Fallback: usar primeras palabras como headline
            text = (chunk.page_content or "")[:100].replace("\n", " ").strip()
            meta["semantic_headline"] = text[:80]
            meta["semantic_summary"] = ""
        chunk.metadata = meta

    print(f"✅ Enriquecimiento completado: {enriched_count}/{len(chunks)} chunks enriquecidos")

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

        # ✅ contains_metrics: presencia de datos cuantitativos (números con %, €, $, o KPIs)
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
    Pipeline HÍBRIDO:
      FASE 1: Chunking determinista por páginas (100% cobertura garantizada)
      FASE 2: Enriquecimiento LLM (solo headline + summary, nunca pierde texto)
      FASE 3: Context injection + almacenamiento en ChromaDB
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

    try:
        print(f"🚀 Ingesta Avanzada (INCREMENTAL) para: {data_path}")

        if not os.path.exists(data_path):
            print(f"❌ data_path no existe: {data_path}")
            return False

        documents = load_documents_from_path(data_path)
        if not documents:
            print("⚠️ No se encontraron documentos válidos.")
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

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings,
        )

        for rel in deleted_files:
            print(f"🗑️ Eliminando del índice: {rel}")
            _delete_by_source_file(vector_store, rel)
            (manifest.get("files") or {}).pop(rel, None)

        if not changed_files and not deleted_files:
            print("✅ No hay cambios detectados. Índice ya actualizado.")
            return True

        changed_set = set(changed_files)

        def _doc_rel(d: Document) -> str:
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("filepath") or meta.get("path") or ""
            if src and os.path.isabs(src) and os.path.exists(src):
                return _get_rel(src, data_path)
            return _safe_norm(meta.get("relative_path") or meta.get("filename") or "")

        docs_to_index = [d for d in documents if _doc_rel(d) in changed_set]

        # Pre-limpieza de texto
        for d in docs_to_index:
            d.page_content = _clean_text_for_embeddings(d.page_content or "")

        print(f"🧾 Cambios detectados: {len(changed_files)} archivo(s) a re-indexar.")
        if not docs_to_index:
            print("⚠️ No se encontraron docs a re-indexar.")
            return False

        for rel in changed_files:
            print(f"♻️ Re-index: limpiando chunks previos de {rel}")
            _delete_by_source_file(vector_store, rel)

        # Detectar contaminación (solo reporta, no modifica)
        _detect_contamination(docs_to_index)

        # Eliminar bloques repetidos entre páginas (ej: slides-template duplicadas)
        docs_to_index = _dedup_repeated_blocks_across_pages(docs_to_index, min_block_words=25)

        # ================================================================
        # FASE 1: Chunking determinista por páginas
        # ================================================================
        print("\n📐 FASE 1: Chunking determinista por páginas...")
        page_chunks = _page_based_chunking(
            docs_to_index,
            min_words=100,
            target_words=250,
            max_words=500,
        )

        if not page_chunks:
            print("⚠️ No se generaron chunks. Abortando.")
            return False

        # ================================================================
        # FASE 1.5: Micro-chunks (child) para Q&A preciso
        # ================================================================
        print("\n🔬 FASE 1.5: Generando micro-chunks (children)...")
        micro_chunks = _generate_micro_chunks(
            page_chunks,
            target_tokens=250,
            overlap_tokens=60,
        )

        # ================================================================
        # FASE 2: Enriquecimiento LLM (headline + summary)
        # ================================================================
        # Enriquecer macro-chunks (parents) con LLM
        if _should_use_semantic_enrichment():
            print("\n🏷️ FASE 2: Enriquecimiento LLM (headline + summary)...")
            try:
                page_chunks = enrich_chunks_with_llm(page_chunks, batch_size=8)
                # Propagar headline/summary del parent a sus micro-chunks
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
                print(f"   ⚠️ Enriquecimiento LLM falló ({e}). Chunks se guardan sin headline/summary.")

        # Guardar semantic chunks para evaluación
        _save_semantic_chunks(page_chunks, base_dir=vector_store_path)

        # Enriquecer metadata (métricas, listas, etc.) en AMBOS niveles
        page_chunks = _enrich_chunk_metadata(page_chunks)
        micro_chunks = _enrich_chunk_metadata(micro_chunks)

        # ================================================================
        # FASE 3: Context injection + almacenamiento
        # ================================================================
        print("\n💾 FASE 3: Context injection + almacenamiento...")

        # Context injection para ambos niveles
        macro_final = inject_context_to_chunks(page_chunks)
        micro_final = inject_context_to_chunks(micro_chunks)

        # Combinar: macro + micro → todos se indexan en ChromaDB
        final_chunks = macro_final + micro_final

        if not final_chunks:
            return False

        # Guardar chunks finales
        if str(os.environ.get("SAVE_FINAL_CHUNKS_JSON", "1")).strip() in ("1", "true", "True", "yes", "YES"):
            _save_final_chunks(final_chunks, base_dir=vector_store_path)

        # Normalizar metadata para ChromaDB
        for ch in final_chunks:
            meta = ch.metadata or {}
            rel = _chunk_source_rel(ch, data_path)
            meta["source_file"] = rel
            meta["relative_path"] = rel
            # ChromaDB no soporta listas — ya son strings desde _merge_pages_into_chunk
            ch.metadata = meta

        # Estadísticas finales
        macro_count = sum(1 for c in final_chunks if (c.metadata or {}).get("chunk_type") == "macro")
        micro_count = sum(1 for c in final_chunks if (c.metadata or {}).get("chunk_type") == "micro")
        word_counts = [len((c.page_content or "").split()) for c in final_chunks]
        macro_words = [len((c.page_content or "").split()) for c in final_chunks if (c.metadata or {}).get("chunk_type") == "macro"]
        micro_words = [len((c.page_content or "").split()) for c in final_chunks if (c.metadata or {}).get("chunk_type") == "micro"]

        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   • Docs entrada: {len(docs_to_index)} páginas")
        print(f"   • Total chunks indexados: {len(final_chunks)}")
        print(f"   • Macro-chunks (parents): {macro_count}")
        if macro_words:
            print(f"     Palabras: min={min(macro_words)}, max={max(macro_words)}, avg={sum(macro_words)/len(macro_words):.0f}")
        print(f"   • Micro-chunks (children): {micro_count}")
        if micro_words:
            print(f"     Palabras: min={min(micro_words)}, max={max(micro_words)}, avg={sum(micro_words)/len(micro_words):.0f}")
        with_headline = sum(1 for c in final_chunks if (c.metadata or {}).get("semantic_headline"))
        print(f"   • Con headline: {with_headline}/{len(final_chunks)}")

        batch_size = 150
        total = len(final_chunks)
        for i in range(0, total, batch_size):
            batch = final_chunks[i: i + batch_size]
            vector_store.add_documents(batch)
            print(f"💾 Guardando lote {i//batch_size + 1}. ({min(i + batch_size, total)}/{total})")

        files_dict = manifest.get("files") or {}
        for rel in changed_files:
            abs_path = current_files.get(rel)
            if abs_path and os.path.exists(abs_path):
                files_dict[rel] = _file_sig(abs_path)
        manifest["files"] = files_dict
        _save_manifest(vector_store_path, manifest)

        print(f"\n✅ Ingesta INCREMENTAL finalizada en: {vector_store_path}")
        return True

    except Exception as e:
        print(f"❌ Error CRÍTICO en ingesta incremental: {e}")
        import traceback
        traceback.print_exc()
        return False