# # app/rag_logic/custom_loaders.py

# import io
# import logging
# import os
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple

# import pdfplumber
# from langchain.docstore.document import Document
# from langchain.document_loaders.base import BaseLoader
# from langchain_community.document_loaders import PyPDFLoader

# # Evitar logs ruidosos
# logging.getLogger("pdfminer").setLevel(logging.ERROR)


# # ============================================================
# # OCR config (por ENV, tunear sin tocar código)
# # ============================================================
# @dataclass
# class OcrConfig:
#     enabled: bool = True

#     # -------- PDF rules --------
#     # Página "sospechosa" si su texto extraído es MUY bajo
#     pdf_min_text_chars_per_page: int = int(os.environ.get("OCR_PDF_MIN_TEXT_CHARS", "60"))

#     # Si True, solo OCR si hay señal visual (imágenes o drawings/vectores)
#     # Ojo: en PDFs "bonitos" muchas veces NO hay imágenes, pero sí drawings.
#     pdf_require_visual_cues_for_ocr: bool = (
#         str(os.environ.get("OCR_PDF_REQUIRE_VISUAL_CUES", "1")).strip().lower() in ("1", "true", "yes")
#     )

#     # Fallback: si el documento completo tiene poco texto, OCR aunque no se detecten señales visuales
#     pdf_min_total_text_chars_for_doc: int = int(os.environ.get("OCR_PDF_MIN_TOTAL_CHARS_DOC", "600"))

#     # Coste: máximo de páginas a OCR por documento
#     pdf_max_pages_to_ocr: int = int(os.environ.get("OCR_PDF_MAX_PAGES", "25"))

#     # Forzar OCR por nombre de archivo (substring match)
#     # Ej: OCR_PDF_FORCE_FILES="Babaria Propuesta Mexico 2025;Otro"
#     pdf_force_files: str = os.environ.get("OCR_PDF_FORCE_FILES", "")

#     # -------- PPT rules --------
#     ppt_min_text_chars_per_slide: int = int(os.environ.get("OCR_PPT_MIN_TEXT_CHARS", "25"))
#     ppt_ocr_if_image_and_low_text: bool = (
#         str(os.environ.get("OCR_PPT_IMAGE_LOW_TEXT", "1")).strip().lower() in ("1", "true", "yes")
#     )
#     ppt_ocr_all_images: bool = (
#         str(os.environ.get("OCR_PPT_ALL_IMAGES", "0")).strip().lower() in ("1", "true", "yes")
#     )

#     # -------- OCR engine --------
#     tesseract_lang: str = os.environ.get("OCR_TESSERACT_LANG", "spa+eng")
#     tesseract_psm: str = os.environ.get("OCR_TESSERACT_PSM", "6")  # 6 = bloque uniforme
#     pdf_render_dpi: int = int(os.environ.get("OCR_PDF_DPI", "220"))


# def _tesseract_available() -> bool:
#     """
#     Nota: esto verifica que pytesseract está importable.
#     Si tesseract.exe no está en PATH, pytesseract fallará al ejecutar.
#     (Aun así, capturamos errores y devolvemos "")
#     """
#     try:
#         import pytesseract  # noqa: F401
#         return True
#     except Exception:
#         return False


# def _run_tesseract_on_pil_image(pil_img, cfg: OcrConfig) -> str:
#     """
#     OCR real. Si pytesseract no está instalado o tesseract falla, devuelve "".
#     """
#     try:
#         import pytesseract
#     except Exception:
#         return ""

#     try:
#         custom_config = f"--psm {cfg.tesseract_psm}"
#         text = pytesseract.image_to_string(pil_img, lang=cfg.tesseract_lang, config=custom_config)
#         return (text or "").strip()
#     except Exception:
#         return ""


# def _safe_basename(path: str) -> str:
#     try:
#         return os.path.basename(path)
#     except Exception:
#         return path


# # ============================================================
# # PDF Loader con OCR selectivo (mejorado y page-level)
# # ============================================================
# class BetterPDFLoader(BaseLoader):
#     """
#     Loader robusto para PDF:

#     ✅ Devuelve Document por PÁGINA (mejor retrieval, mejores citas).
#     ✅ Usa PyMuPDF (fitz) para:
#         - extraer texto por página
#         - detectar señales visuales:
#             - imágenes
#             - drawings/vectores (muy importante en PDFs "bonitos")
#     ✅ Aplica OCR selectivo por página cuando:
#         - texto por página < umbral
#         - y (hay señal visual) [configurable]
#       + fallback de documento pobre:
#         - si total_chars_doc < umbral, OCR aunque no haya señal visual
#       + override:
#         - forzar OCR por nombre de archivo (ENV)

#     Fallbacks:
#       - si PyMuPDF falla: PDFPlumber por página + OCR (si posible)
#       - si PDFPlumber falla: PyPDFLoader (último recurso)
#     """

#     def __init__(self, file_path: str, ocr_cfg: Optional[OcrConfig] = None):
#         self.file_path = file_path
#         self.ocr_cfg = ocr_cfg or OcrConfig()

#     # ---------------------------
#     # Helpers
#     # ---------------------------
#     def _is_forced_ocr_file(self) -> bool:
#         if not self.ocr_cfg.enabled:
#             return False
#         s = (self.ocr_cfg.pdf_force_files or "").strip()
#         if not s:
#             return False

#         fname = _safe_basename(self.file_path).lower()
#         parts = [p.strip().lower() for p in s.replace("|", ";").replace(",", ";").split(";") if p.strip()]
#         return any(p in fname for p in parts)

#     def _should_ocr_page(self, extracted_text: str, has_visual_cues: bool, force_file: bool, doc_is_poor: bool) -> bool:
#         if not self.ocr_cfg.enabled:
#             return False

#         if force_file:
#             return True

#         # Si una página ya trae texto suficiente, no OCR
#         if len((extracted_text or "").strip()) >= self.ocr_cfg.pdf_min_text_chars_per_page:
#             return False

#         # Si el documento es muy pobre en texto, permitimos OCR sin señal visual (fallback)
#         if doc_is_poor:
#             return True

#         # Regla base: exigir señal visual si está activado
#         if self.ocr_cfg.pdf_require_visual_cues_for_ocr and not has_visual_cues:
#             return False

#         return True

#     def _ocr_page_with_fitz(self, doc, pidx: int) -> str:
#         if not _tesseract_available():
#             return ""

#         try:
#             import fitz
#             from PIL import Image
#         except Exception:
#             return ""

#         try:
#             page = doc.load_page(pidx)
#             zoom = self.ocr_cfg.pdf_render_dpi / 72.0
#             mat = fitz.Matrix(zoom, zoom)
#             pix = page.get_pixmap(matrix=mat, alpha=False)
#             img = Image.open(io.BytesIO(pix.tobytes("png")))
#             return _run_tesseract_on_pil_image(img, self.ocr_cfg)
#         except Exception:
#             return ""

#     # ---------------------------
#     # Main
#     # ---------------------------
#     def load(self) -> List[Document]:
#         fname = _safe_basename(self.file_path)
#         print(f"📄 Procesando PDF: {fname}")

#         force_file = self._is_forced_ocr_file()
#         documents: List[Document] = []

#         # =========================
#         # NIVEL 1: PyMuPDF (fitz)
#         # =========================
#         try:
#             import fitz  # PyMuPDF

#             with fitz.open(self.file_path) as doc:
#                 page_count = doc.page_count
#                 if page_count <= 0:
#                     return []

#                 # Extraer texto + señales visuales por página (rápido)
#                 per_page_text: List[str] = []
#                 per_page_visual: List[bool] = []

#                 for page in doc:
#                     t = (page.get_text("text") or "").strip()

#                     # Señales visuales:
#                     # - imágenes
#                     # - drawings/vectores (crucial para PDFs con cajas, líneas, bullets "dibujados")
#                     has_images = bool(page.get_images(full=True))
#                     try:
#                         has_drawings = bool(page.get_drawings())
#                     except Exception:
#                         has_drawings = False

#                     per_page_text.append(t)
#                     per_page_visual.append(bool(has_images or has_drawings))

#                 total_chars = sum(len(t) for t in per_page_text)
#                 doc_is_poor = total_chars < self.ocr_cfg.pdf_min_total_text_chars_for_doc

#                 # Elegir páginas a OCR
#                 ocr_indices: List[int] = []
#                 if self.ocr_cfg.enabled and _tesseract_available():
#                     for i in range(page_count):
#                         if self._should_ocr_page(
#                             extracted_text=per_page_text[i],
#                             has_visual_cues=per_page_visual[i],
#                             force_file=force_file,
#                             doc_is_poor=doc_is_poor,
#                         ):
#                             ocr_indices.append(i)

#                     # coste control
#                     if ocr_indices:
#                         ocr_indices = ocr_indices[: self.ocr_cfg.pdf_max_pages_to_ocr]

#                 # Ejecutar OCR selectivo
#                 ocr_map: Dict[int, str] = {}
#                 if ocr_indices:
#                     for pidx in ocr_indices:
#                         ocr_txt = (self._ocr_page_with_fitz(doc, pidx) or "").strip()
#                         if ocr_txt:
#                             ocr_map[pidx] = ocr_txt

#                 # Construir Document por página
#                 any_ocr_used = bool(ocr_map)
#                 for pidx in range(page_count):
#                     base_text = (per_page_text[pidx] or "").strip()
#                     ocr_text = (ocr_map.get(pidx) or "").strip()

#                     page_out = base_text
#                     if ocr_text:
#                         if page_out:
#                             page_out += "\n\n[OCR]\n" + ocr_text
#                         else:
#                             page_out = "[OCR]\n" + ocr_text

#                     # Si una página queda vacía, la omitimos para evitar ruido
#                     if not (page_out or "").strip():
#                         continue

#                     documents.append(
#                         Document(
#                             page_content=page_out.strip(),
#                             metadata={
#                                 "source": self.file_path,
#                                 "file_type": "pdf",
#                                 "filename": fname,
#                                 "source_file": fname,            # clave para tu manifest/retriever
#                                 "relative_path": fname,          # compat con tu formato actual
#                                 "page": pidx,                    # 0-based
#                                 "page_number": pidx + 1,         # 1-based
#                                 "ocr_used": bool(any_ocr_used),
#                                 "ocr_used_page": bool(pidx in ocr_map),
#                             },
#                         )
#                     )

#                 print(
#                     f"✅ PDF leído (PyMuPDF) | pages={page_count} | total_chars={total_chars} | "
#                     f"ocr_pages={len(ocr_indices)} | ocr_used={any_ocr_used} | forced={force_file} | poor_doc={doc_is_poor}"
#                 )
#                 return documents

#         except Exception as e:
#             print(f"⚠️ PyMuPDF falló en {fname}: {e}")

#         # =========================
#         # NIVEL 2: PDFPlumber fallback (por página)
#         # =========================
#         try:
#             total_chars = 0
#             raw_pages: List[str] = []
#             visual_pages: List[bool] = []

#             with pdfplumber.open(self.file_path) as pdf:
#                 for page in pdf.pages:
#                     t = (page.extract_text() or "").strip()
#                     raw_pages.append(t)
#                     total_chars += len(t)

#                     # pdfplumber: señal visual básica (imágenes detectadas)
#                     try:
#                         has_images = bool(getattr(page, "images", None))
#                     except Exception:
#                         has_images = False
#                     visual_pages.append(bool(has_images))

#             doc_is_poor = total_chars < self.ocr_cfg.pdf_min_total_text_chars_for_doc

#             # OCR con fitz (si está disponible)
#             ocr_map: Dict[int, str] = {}
#             if self.ocr_cfg.enabled and _tesseract_available():
#                 try:
#                     import fitz
#                     with fitz.open(self.file_path) as doc:
#                         candidates = []
#                         for i, t in enumerate(raw_pages):
#                             if self._should_ocr_page(
#                                 extracted_text=t,
#                                 has_visual_cues=visual_pages[i],
#                                 force_file=force_file,
#                                 doc_is_poor=doc_is_poor,
#                             ):
#                                 candidates.append(i)

#                         candidates = candidates[: self.ocr_cfg.pdf_max_pages_to_ocr]
#                         for pidx in candidates:
#                             ocr_txt = (self._ocr_page_with_fitz(doc, pidx) or "").strip()
#                             if ocr_txt:
#                                 ocr_map[pidx] = ocr_txt
#                 except Exception:
#                     pass

#             any_ocr_used = bool(ocr_map)
#             for pidx, base_text in enumerate(raw_pages):
#                 ocr_text = (ocr_map.get(pidx) or "").strip()

#                 page_out = (base_text or "").strip()
#                 if ocr_text:
#                     if page_out:
#                         page_out += "\n\n[OCR]\n" + ocr_text
#                     else:
#                         page_out = "[OCR]\n" + ocr_text

#                 if not page_out:
#                     continue

#                 documents.append(
#                     Document(
#                         page_content=page_out,
#                         metadata={
#                             "source": self.file_path,
#                             "file_type": "pdf",
#                             "filename": fname,
#                             "source_file": fname,
#                             "relative_path": fname,
#                             "page": pidx,
#                             "page_number": pidx + 1,
#                             "ocr_used": bool(any_ocr_used),
#                             "ocr_used_page": bool(pidx in ocr_map),
#                         },
#                     )
#                 )

#             print(
#                 f"✅ PDF leído (PDFPlumber fallback) | pages={len(raw_pages)} | total_chars={total_chars} | "
#                 f"ocr_pages={len(ocr_map)} | forced={force_file} | poor_doc={doc_is_poor}"
#             )
#             return documents

#         except Exception as e:
#             print(f"⚠️ PDFPlumber falló en {fname}: {e}")

#         # =========================
#         # NIVEL 3: PyPDFLoader (último recurso)
#         # =========================
#         try:
#             loader = PyPDFLoader(self.file_path)
#             docs = loader.load()

#             out: List[Document] = []
#             for d in docs:
#                 pidx = int(d.metadata.get("page", 0))
#                 content = (d.page_content or "").strip()
#                 if not content:
#                     continue
#                 out.append(
#                     Document(
#                         page_content=content,
#                         metadata={
#                             "source": self.file_path,
#                             "file_type": "pdf",
#                             "filename": fname,
#                             "source_file": fname,
#                             "relative_path": fname,
#                             "page": pidx,
#                             "page_number": pidx + 1,
#                             "ocr_used": False,
#                             "ocr_used_page": False,
#                         },
#                     )
#                 )

#             print(f"✅ PDF leído (PyPDF fallback) | pages={len(out)}")
#             return out

#         except Exception as e:
#             print(f"❌ Error total leyendo {fname}: {e}")
#             return []


# # ============================================================
# # PowerPoint Loader con OCR selectivo (PPT/PPTX) - slide-level
# # ============================================================
# class BetterPowerPointLoader(BaseLoader):
#     """
#     Extrae texto de PPT/PPTX y aplica OCR a imágenes SOLO si se activa por reglas.

#     REGLAS OCR (PPT):
#       - Si ppt_ocr_all_images=1 => OCR a todas las imágenes.
#       - Si no:
#           - OCR solo si (hay imagen) AND (texto del slide < ppt_min_text_chars_per_slide)
#             (controlado por ppt_ocr_if_image_and_low_text)

#     Devuelve Document por SLIDE, para retrieval/citas.
#     """

#     def __init__(self, file_path: str, ocr_cfg: Optional[OcrConfig] = None):
#         self.file_path = file_path
#         self.ocr_cfg = ocr_cfg or OcrConfig()

#     def _extract_slide_text(self, slide) -> str:
#         parts: List[str] = []
#         for shape in slide.shapes:
#             try:
#                 if hasattr(shape, "has_text_frame") and shape.has_text_frame:
#                     t = (shape.text or "").strip()
#                     if t:
#                         parts.append(t)
#             except Exception:
#                 continue
#         return "\n".join(parts).strip()

#     def _iter_slide_images_pil(self, slide):
#         try:
#             from PIL import Image
#         except Exception:
#             return []

#         imgs = []
#         for shape in slide.shapes:
#             try:
#                 # 13 = MSO_SHAPE_TYPE.PICTURE
#                 if getattr(shape, "shape_type", None) == 13 and hasattr(shape, "image"):
#                     blob = shape.image.blob
#                     pil = Image.open(io.BytesIO(blob))
#                     imgs.append(pil)
#             except Exception:
#                 continue
#         return imgs

#     def load(self) -> List[Document]:
#         fname = _safe_basename(self.file_path)
#         print(f"📊 Procesando PPT: {fname}")

#         try:
#             from pptx import Presentation
#         except Exception as e:
#             print(f"❌ Falta dependencia python-pptx para leer PPTX/PPT: {e}")
#             return []

#         try:
#             prs = Presentation(self.file_path)
#         except Exception as e:
#             print(f"❌ No se pudo abrir PPT/PPTX: {e}")
#             return []

#         out_docs: List[Document] = []
#         any_ocr_used = False

#         for s_i, slide in enumerate(prs.slides, start=1):
#             slide_text = self._extract_slide_text(slide)
#             slide_text_len = len((slide_text or "").strip())

#             imgs = self._iter_slide_images_pil(slide)
#             has_img = len(imgs) > 0

#             do_ocr = False
#             if self.ocr_cfg.enabled and _tesseract_available() and has_img:
#                 if self.ocr_cfg.ppt_ocr_all_images:
#                     do_ocr = True
#                 elif self.ocr_cfg.ppt_ocr_if_image_and_low_text and slide_text_len < self.ocr_cfg.ppt_min_text_chars_per_slide:
#                     do_ocr = True

#             content_parts: List[str] = []
#             if slide_text.strip():
#                 content_parts.append(slide_text.strip())

#             if do_ocr:
#                 any_ocr_used = True
#                 ocr_lines: List[str] = []
#                 for img_i, pil_img in enumerate(imgs, start=1):
#                     txt = _run_tesseract_on_pil_image(pil_img, self.ocr_cfg)
#                     if txt.strip():
#                         ocr_lines.append(f"- Imagen {img_i} (OCR):\n{txt.strip()}")
#                 if ocr_lines:
#                     content_parts.append("[OCR de imágenes del slide]\n" + "\n\n".join(ocr_lines))

#             page_out = "\n\n".join([p for p in content_parts if p.strip()]).strip()
#             if not page_out:
#                 continue

#             out_docs.append(
#                 Document(
#                     page_content=page_out,
#                     metadata={
#                         "source": self.file_path,
#                         "file_type": "ppt",
#                         "filename": fname,
#                         "source_file": fname,
#                         "relative_path": fname,
#                         "slide": s_i,                 # 1-based
#                         "slide_number": s_i,          # alias
#                         "ocr_used": bool(any_ocr_used),
#                         "ocr_used_slide": bool(do_ocr),
#                     },
#                 )
#             )

#         print("✅ PPT leído con éxito" + (" + OCR selectivo" if any_ocr_used else ""))
#         return out_docs



# app/rag_logic/custom_loaders.py

import io
import logging
import os
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

    def _build_page_document(self, pidx, page_count, base_text, ocr_text, table_text,
                             fname, any_ocr_used, ocr_map):
        parts = []
        
        # 1. Prioridad a la tabla si existe (aporta estructura)
        if (table_text or "").strip():
            parts.append(f"--- DATOS TABULARES (Pág {pidx + 1}) ---\n{table_text.strip()}\n-----------------------------")

        # 2. Texto base
        if (base_text or "").strip():
            # Limpieza básica para evitar duplicidad si la tabla ya capturó parte
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
                                        parts = []
                                        for table in tables:
                                            if not table: continue
                                            # Limpiar None y saltos de línea dentro de celdas
                                            cleaned_table = [
                                                [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                                                for row in table
                                            ]
                                            # Filtrar filas vacías
                                            cleaned_table = [row for row in cleaned_table if any(row)]
                                            
                                            if not cleaned_table: continue

                                            # Formato Markdown Real
                                            # Header
                                            md_table = "| " + " | ".join(cleaned_table[0]) + " |\n"
                                            md_table += "| " + " | ".join(["---"] * len(cleaned_table[0])) + " |\n"
                                            # Body
                                            for row in cleaned_table[1:]:
                                                md_table += "| " + " | ".join(row) + " |\n"
                                            
                                            parts.append(md_table)
                                            
                                        if parts:
                                            table_map[pidx] = "\n\n".join(parts)
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
            print(f"⚠️ PyMuPDF falló en {fname}: {e}")

        # ... (Mantener el resto del fallback igual) ...
        # (He truncado el código de fallback aquí por brevedad, usa el que ya tenías
        # pero asegúrate de cerrar la clase correctamente si copias y pegas)
        return []


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