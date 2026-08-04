import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pdfplumber
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain_community.document_loaders import PyPDFLoader

logging.getLogger("pdfminer").setLevel(logging.ERROR)


# ============================================================
# OCR config
# ============================================================
@dataclass
class LoaderConfig:
    """
    Ajustes de la carga de documentos.

    Aquí hubo OCR: detección de páginas pobres, señales visuales, render a DPI
    y tesseract. Se eliminó el 4-ago-2026 porque NO PODÍA EJECUTARSE: ni
    `pytesseract` está en requirements-prod.txt ni `tesseract` en ninguno de los
    dos Dockerfiles, así que `_tesseract_available()` devolvía False siempre y
    toda esa rama era inalcanzable en la imagen desplegada.

    Dejarla habría sido mantener —y describir— una capacidad que el sistema no
    tiene. Si algún día el corpus trae escaneados sin capa de texto, se añade la
    dependencia y se vuelve a escribir; hoy los PDFs de este corpus llevan texto.
    """

    pdf_extract_tables: bool = (
        str(os.environ.get("OCR_PDF_EXTRACT_TABLES", "1")).strip().lower() in ("1", "true", "yes")
    )


def _safe_basename(path: str) -> str:
    try:
        return os.path.basename(path)
    except Exception:
        return path


# ============================================================
# PDF Loader — NUNCA descarta páginas
# ============================================================
class BetterPDFLoader(BaseLoader):
    """
    Loader robusto para PDF:
    ✅ Devuelve 1 Document por PÁGINA (NUNCA descarta).
    ✅ Combina: texto (fitz, layout-aware) + tablas en Markdown (pdfplumber).
    ✅ Páginas vacías se marcan con is_empty_page=True.
    """

    def __init__(self, file_path: str, loader_cfg: Optional[LoaderConfig] = None):
        self.file_path = file_path
        self.loader_cfg = loader_cfg or LoaderConfig()




    @staticmethod
    def _tables_to_markdown(tables) -> str:
        parts = []
        for table in tables:
            if not table:
                continue
            cleaned_table = [
                [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                for row in table
            ]
            cleaned_table = [row for row in cleaned_table if any(row)]
            if not cleaned_table:
                continue
            md_table = "| " + " | ".join(cleaned_table[0]) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(cleaned_table[0])) + " |\n"
            for row in cleaned_table[1:]:
                md_table += "| " + " | ".join(row) + " |\n"
            parts.append(md_table)
        return "\n\n".join(parts)

    def _load_with_pdfplumber(self) -> List[Document]:
        """
        Fallback cuando PyMuPDF (fitz) no está disponible o falla.
        Con texto y tablas reales — nunca debe devolver [] salvo que el PDF en
        sí sea ilegible (corrupto o cifrado).
        """
        fname = _safe_basename(self.file_path)
        try:
            with pdfplumber.open(self.file_path) as pdf:
                page_count = len(pdf.pages)
                if page_count <= 0:
                    return []
                documents: List[Document] = []
                empty_pages = []
                for pidx, page in enumerate(pdf.pages):
                    base_text = (page.extract_text() or "").strip()
                    table_text = ""
                    if self.loader_cfg.pdf_extract_tables:
                        try:
                            tables = page.extract_tables()
                            if tables:
                                table_text = self._tables_to_markdown(tables)
                        except Exception:
                            pass
                    doc_page = self._build_page_document(
                        pidx, page_count, base_text, table_text, fname,
                    )
                    if doc_page.metadata.get("is_empty_page"):
                        empty_pages.append(pidx + 1)
                    documents.append(doc_page)

                print(
                    f"✅ PDF leído (pdfplumber fallback) | total_pages={page_count} | "
                    f"pages_with_text={page_count - len(empty_pages)} | empty_pages={len(empty_pages)}"
                )
                if empty_pages:
                    print(f"   📋 Páginas vacías: {empty_pages}")
                return documents
        except Exception as e:
            print(f"❌ pdfplumber también falló en {fname}: {e}")
            return []

    @staticmethod
    def _strip_table_lines(base_text: str, table_text: str) -> str:
        """
        Quita del texto plano las líneas que la tabla ya recoge en markdown.

        El extractor de texto (fitz/pdfplumber) devuelve la tabla como palabras
        sueltas, y pdfplumber la devuelve otra vez como markdown. Sin esto, la
        misma fila viaja DOS veces en el mismo chunk:

            | 4.5 - 5.0 | Outstanding | Top 10%. Exceptional impact... |
            4.5 - 5.0 Outstanding Top 10%. Exceptional impact...

        Medido el 3-ago-2026 antes de este arreglo: 11 de los 14 chunks con
        tabla repetían celdas, y el 56% de las líneas largas del índice entero
        eran repeticiones. Se paga en tokens de prompt en CADA consulta y
        distorsiona las frecuencias de BM25.

        Criterio deliberadamente conservador: solo se borra una línea si su
        forma normalizada coincide EXACTAMENTE con una fila completa de la tabla
        o con una celda de cierta longitud. Ante la duda, se conserva: perder
        contenido real es mucho peor que dejar una repetición.
        """
        if not (base_text or "").strip() or not (table_text or "").strip():
            return base_text

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s or "").strip().lower()

        prohibidas = set()
        for linea in table_text.splitlines():
            if not linea.strip().startswith("|"):
                continue
            celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
            if all(set(c) <= set("- ") for c in celdas):      # fila separadora
                continue
            fila = norm(" ".join(c for c in celdas if c))
            if fila:
                prohibidas.add(fila)
            for c in celdas:
                if len(c) >= 12:      # celdas cortas ("25", "Sí") pueden ser texto legítimo
                    prohibidas.add(norm(c))

        conservadas = [ln for ln in base_text.splitlines() if norm(ln) not in prohibidas]
        # Cada línea borrada deja un hueco. Sin colapsarlos, quitar tres filas
        # seguidas deja seis saltos de línea que se pagan en tokens de prompt.
        return re.sub(r"\n{3,}", "\n\n", "\n".join(conservadas)).strip()

    def _build_page_document(self, pidx, page_count, base_text, table_text, fname):
        parts = []

        # 1. Prioridad a la tabla si existe (aporta estructura)
        if (table_text or "").strip():
            parts.append(f"--- DATOS TABULARES (Pág {pidx + 1}) ---\n{table_text.strip()}\n-----------------------------")

        # 2. Texto base, ya sin las líneas que la tabla acaba de recoger
        base_text = self._strip_table_lines(base_text, table_text)
        if (base_text or "").strip():
            parts.append(base_text.strip())

        page_out = "\n\n".join(parts).strip()
        is_empty = not page_out or len(page_out.split()) < 3
        if is_empty:
            page_out = f"[Página {pidx + 1} — contenido visual sin texto extraíble]"

        return Document(
            page_content=page_out,
            metadata={
                "source": self.file_path, "file_type": "pdf",
                "filename": fname, "source_file": fname, "relative_path": fname,
                "page": pidx, "page_number": pidx + 1, "page_count": page_count,
                "has_table": bool((table_text or "").strip()),
                "is_empty_page": is_empty, "text_chars": len(page_out),
            },
        )

    def load(self) -> List[Document]:
        fname = _safe_basename(self.file_path)
        print(f"📄 Procesando PDF: {fname}")
        documents: List[Document] = []

        # === NIVEL 1: PyMuPDF ===
        try:
            import fitz
            with fitz.open(self.file_path) as doc:
                page_count = doc.page_count
                if page_count <= 0:
                    return []
                per_page_text, per_page_visual = [], []
                for page in doc:
                    # MEJORA: sort=True para respetar columnas y layout visual
                    t = (page.get_text("text", sort=True) or "").strip()
                    
                    has_images = bool(page.get_images(full=True))
                    try:
                        has_drawings = bool(page.get_drawings())
                    except Exception:
                        has_drawings = False
                    per_page_text.append(t)
                    per_page_visual.append(bool(has_images or has_drawings))

                total_chars = sum(len(t) for t in per_page_text)

                # MEJORA: Extracción de tablas a Markdown
                table_map = {}
                if self.loader_cfg.pdf_extract_tables:
                    try:
                        with pdfplumber.open(self.file_path) as plumber_pdf:
                            for pidx in range(min(page_count, len(plumber_pdf.pages))):
                                try:
                                    tables = plumber_pdf.pages[pidx].extract_tables()
                                    if tables:
                                        md = self._tables_to_markdown(tables)
                                        if md:
                                            table_map[pidx] = md
                                except Exception:
                                    continue
                    except Exception as e:
                        print(f"   ⚠️ Extracción de tablas falló: {e}")

                empty_pages = []
                for pidx in range(page_count):
                    doc_page = self._build_page_document(
                        pidx, page_count, per_page_text[pidx],
                        table_map.get(pidx, ""), fname,
                    )
                    if doc_page.metadata.get("is_empty_page"):
                        empty_pages.append(pidx + 1)
                    documents.append(doc_page)

                print(
                    f"✅ PDF leído (PyMuPDF Enhanced) | total_pages={page_count} | "
                    f"pages_with_text={page_count - len(empty_pages)} | pages_with_tables={len(table_map)} | "
                    f"empty_pages={len(empty_pages)} | total_chars={total_chars}"
                )
                if empty_pages:
                    print(f"   📋 Páginas vacías: {empty_pages}")
                return documents
        except Exception as e:
            print(f"⚠️ PyMuPDF falló en {fname}: {e} — usando fallback pdfplumber")

        # === NIVEL 2: pdfplumber — nunca devolver [] solo por falta de fitz ===
        return self._load_with_pdfplumber()


# ============================================================
# PowerPoint Loader — NUNCA descarta slides
# ============================================================
class BetterPowerPointLoader(BaseLoader):
    def __init__(self, file_path: str, loader_cfg: Optional[LoaderConfig] = None):
        self.file_path = file_path
        self.loader_cfg = loader_cfg or LoaderConfig()

    def _extract_slide_text(self, slide):
        parts = []
        for shape in slide.shapes:
            try:
                if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                    t = (shape.text or "").strip()
                    if t:
                        parts.append(t)
            except Exception:
                continue
        return "\n".join(parts).strip()


    def load(self):
        fname = _safe_basename(self.file_path)
        print(f"📊 Procesando PPT: {fname}")
        try:
            from pptx import Presentation
        except Exception as e:
            print(f"❌ Falta python-pptx: {e}")
            return []
        try:
            prs = Presentation(self.file_path)
        except Exception as e:
            print(f"❌ No se pudo abrir PPT: {e}")
            return []

        out_docs = []
        slide_count = len(prs.slides)
        for s_i, slide in enumerate(prs.slides, start=1):
            slide_text = self._extract_slide_text(slide)
            page_out = slide_text.strip()
            is_empty = not page_out or len(page_out.split()) < 3
            if is_empty:
                page_out = f"[Slide {s_i} — contenido visual sin texto extraíble]"
            out_docs.append(Document(page_content=page_out, metadata={
                "source": self.file_path, "file_type": "ppt",
                "filename": fname, "source_file": fname, "relative_path": fname,
                "slide": s_i, "slide_number": s_i, "slide_count": slide_count,
                "is_empty_slide": is_empty, "text_chars": len(page_out),
            }))
        print("✅ PPT leído")
        return out_docs