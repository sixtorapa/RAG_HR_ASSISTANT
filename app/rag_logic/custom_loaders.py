import io
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
class OcrConfig:
    enabled: bool = True
    pdf_min_text_chars_per_page: int = int(os.environ.get("OCR_PDF_MIN_TEXT_CHARS", "60"))
    pdf_require_visual_cues_for_ocr: bool = (
        str(os.environ.get("OCR_PDF_REQUIRE_VISUAL_CUES", "1")).strip().lower() in ("1", "true", "yes")
    )
    pdf_min_total_text_chars_for_doc: int = int(os.environ.get("OCR_PDF_MIN_TOTAL_CHARS_DOC", "600"))
    pdf_max_pages_to_ocr: int = int(os.environ.get("OCR_PDF_MAX_PAGES", "40"))
    pdf_force_files: str = os.environ.get("OCR_PDF_FORCE_FILES", "")
    ppt_min_text_chars_per_slide: int = int(os.environ.get("OCR_PPT_MIN_TEXT_CHARS", "25"))
    ppt_ocr_if_image_and_low_text: bool = (
        str(os.environ.get("OCR_PPT_IMAGE_LOW_TEXT", "1")).strip().lower() in ("1", "true", "yes")
    )
    ppt_ocr_all_images: bool = (
        str(os.environ.get("OCR_PPT_ALL_IMAGES", "0")).strip().lower() in ("1", "true", "yes")
    )
    tesseract_lang: str = os.environ.get("OCR_TESSERACT_LANG", "spa+eng")
    tesseract_psm: str = os.environ.get("OCR_TESSERACT_PSM", "6")
    pdf_render_dpi: int = int(os.environ.get("OCR_PDF_DPI", "250"))
    pdf_extract_tables: bool = (
        str(os.environ.get("OCR_PDF_EXTRACT_TABLES", "1")).strip().lower() in ("1", "true", "yes")
    )


def _tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        return True
    except Exception:
        return False


def _run_tesseract_on_pil_image(pil_img, cfg: OcrConfig) -> str:
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        custom_config = f"--psm {cfg.tesseract_psm}"
        text = pytesseract.image_to_string(pil_img, lang=cfg.tesseract_lang, config=custom_config)
        return (text or "").strip()
    except Exception:
        return ""


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
    ✅ Combina: texto (fitz layout-aware) + tablas Markdown (pdfplumber) + OCR (tesseract).
    ✅ Páginas vacías se marcan con is_empty_page=True.
    """

    def __init__(self, file_path: str, ocr_cfg: Optional[OcrConfig] = None):
        self.file_path = file_path
        self.ocr_cfg = ocr_cfg or OcrConfig()

    def _is_forced_ocr_file(self) -> bool:
        if not self.ocr_cfg.enabled:
            return False
        s = (self.ocr_cfg.pdf_force_files or "").strip()
        if not s:
            return False
        fname = _safe_basename(self.file_path).lower()
        parts = [p.strip().lower() for p in s.replace("|", ";").replace(",", ";").split(";") if p.strip()]
        return any(p in fname for p in parts)

    def _should_ocr_page(self, extracted_text, has_visual_cues, force_file, doc_is_poor):
        if not self.ocr_cfg.enabled:
            return False
        if force_file:
            return True
        if len((extracted_text or "").strip()) >= self.ocr_cfg.pdf_min_text_chars_per_page:
            return False
        if doc_is_poor:
            return True
        if self.ocr_cfg.pdf_require_visual_cues_for_ocr and not has_visual_cues:
            return False
        return True

    def _ocr_page_with_fitz(self, doc, pidx):
        if not _tesseract_available():
            return ""
        try:
            import fitz
            from PIL import Image
        except Exception:
            return ""
        try:
            page = doc.load_page(pidx)
            zoom = self.ocr_cfg.pdf_render_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return _run_tesseract_on_pil_image(img, self.ocr_cfg)
        except Exception:
            return ""

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
        Sin OCR (el render a imagen para tesseract depende de fitz) pero con
        texto + tablas reales — nunca debe devolver [] salvo que el PDF en sí
        sea ilegible (corrupto/cifrado).
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
                    if self.ocr_cfg.pdf_extract_tables:
                        try:
                            tables = page.extract_tables()
                            if tables:
                                table_text = self._tables_to_markdown(tables)
                        except Exception:
                            pass
                    doc_page = self._build_page_document(
                        pidx, page_count, base_text, "", table_text,
                        fname, False, {},
                    )
                    if doc_page.metadata.get("is_empty_page"):
                        empty_pages.append(pidx + 1)
                    documents.append(doc_page)

                print(
                    f"✅ PDF leído (pdfplumber fallback, sin OCR) | total_pages={page_count} | "
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

    def _build_page_document(self, pidx, page_count, base_text, ocr_text, table_text,
                             fname, any_ocr_used, ocr_map):
        parts = []

        # 1. Prioridad a la tabla si existe (aporta estructura)
        if (table_text or "").strip():
            parts.append(f"--- DATOS TABULARES (Pág {pidx + 1}) ---\n{table_text.strip()}\n-----------------------------")

        # 2. Texto base, ya sin las líneas que la tabla acaba de recoger
        base_text = self._strip_table_lines(base_text, table_text)
        if (base_text or "").strip():
            parts.append(base_text.strip())

        # 3. OCR si fue necesario
        if (ocr_text or "").strip():
            parts.append(f"[TEXTO ADICIONAL OCR]\n{ocr_text.strip()}")

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
                "ocr_used": bool(any_ocr_used), "ocr_used_page": bool(pidx in ocr_map),
                "has_table": bool((table_text or "").strip()),
                "is_empty_page": is_empty, "text_chars": len(page_out),
            },
        )

    def load(self) -> List[Document]:
        fname = _safe_basename(self.file_path)
        print(f"📄 Procesando PDF: {fname}")
        force_file = self._is_forced_ocr_file()
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
                doc_is_poor = total_chars < self.ocr_cfg.pdf_min_total_text_chars_for_doc

                ocr_indices = []
                if self.ocr_cfg.enabled and _tesseract_available():
                    for i in range(page_count):
                        if self._should_ocr_page(per_page_text[i], per_page_visual[i], force_file, doc_is_poor):
                            ocr_indices.append(i)
                    ocr_indices = ocr_indices[: self.ocr_cfg.pdf_max_pages_to_ocr]

                ocr_map = {}
                for pidx in ocr_indices:
                    ocr_txt = (self._ocr_page_with_fitz(doc, pidx) or "").strip()
                    if ocr_txt:
                        ocr_map[pidx] = ocr_txt

                # MEJORA: Extracción de tablas a Markdown
                table_map = {}
                if self.ocr_cfg.pdf_extract_tables:
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

                any_ocr_used = bool(ocr_map)
                empty_pages = []
                for pidx in range(page_count):
                    doc_page = self._build_page_document(
                        pidx, page_count, per_page_text[pidx],
                        ocr_map.get(pidx, ""), table_map.get(pidx, ""),
                        fname, any_ocr_used, ocr_map,
                    )
                    if doc_page.metadata.get("is_empty_page"):
                        empty_pages.append(pidx + 1)
                    documents.append(doc_page)

                print(
                    f"✅ PDF leído (PyMuPDF Enhanced) | total_pages={page_count} | "
                    f"pages_with_text={page_count - len(empty_pages)} | pages_with_tables={len(table_map)} | "
                    f"ocr_pages={len(ocr_map)} | empty_pages={len(empty_pages)} | "
                    f"total_chars={total_chars} | forced={force_file} | poor_doc={doc_is_poor}"
                )
                if empty_pages:
                    print(f"   📋 Páginas vacías: {empty_pages}")
                return documents
        except Exception as e:
            print(f"⚠️ PyMuPDF falló en {fname}: {e} — usando fallback pdfplumber (sin OCR)")

        # === NIVEL 2: pdfplumber (sin OCR) — nunca devolver [] solo por falta de fitz ===
        return self._load_with_pdfplumber()


# ============================================================
# PowerPoint Loader — NUNCA descarta slides
# ============================================================
class BetterPowerPointLoader(BaseLoader):
    def __init__(self, file_path: str, ocr_cfg: Optional[OcrConfig] = None):
        self.file_path = file_path
        self.ocr_cfg = ocr_cfg or OcrConfig()

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

    def _iter_slide_images_pil(self, slide):
        try:
            from PIL import Image
        except Exception:
            return []
        imgs = []
        for shape in slide.shapes:
            try:
                if getattr(shape, "shape_type", None) == 13 and hasattr(shape, "image"):
                    imgs.append(Image.open(io.BytesIO(shape.image.blob)))
            except Exception:
                continue
        return imgs

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

        out_docs, any_ocr_used = [], False
        slide_count = len(prs.slides)
        for s_i, slide in enumerate(prs.slides, start=1):
            slide_text = self._extract_slide_text(slide)
            imgs = self._iter_slide_images_pil(slide)
            do_ocr = False
            if self.ocr_cfg.enabled and _tesseract_available() and imgs:
                if self.ocr_cfg.ppt_ocr_all_images:
                    do_ocr = True
                elif self.ocr_cfg.ppt_ocr_if_image_and_low_text and len(slide_text) < self.ocr_cfg.ppt_min_text_chars_per_slide:
                    do_ocr = True

            parts = []
            if slide_text.strip():
                parts.append(slide_text.strip())
            if do_ocr:
                any_ocr_used = True
                for img_i, pil_img in enumerate(imgs, start=1):
                    txt = _run_tesseract_on_pil_image(pil_img, self.ocr_cfg)
                    if txt.strip():
                        parts.append(f"[OCR imagen {img_i}]\n{txt.strip()}")

            page_out = "\n\n".join(p for p in parts if p.strip()).strip()
            is_empty = not page_out or len(page_out.split()) < 3
            if is_empty:
                page_out = f"[Slide {s_i} — contenido visual sin texto extraíble]"
            out_docs.append(Document(page_content=page_out, metadata={
                "source": self.file_path, "file_type": "ppt",
                "filename": fname, "source_file": fname, "relative_path": fname,
                "slide": s_i, "slide_number": s_i, "slide_count": slide_count,
                "ocr_used": bool(any_ocr_used), "ocr_used_slide": bool(do_ocr),
                "is_empty_slide": is_empty, "text_chars": len(page_out),
            }))
        print("✅ PPT leído" + (" + OCR selectivo" if any_ocr_used else ""))
        return out_docs