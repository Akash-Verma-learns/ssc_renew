"""
proposal_ocr.py
===============
OCR layer for TQ Extractor — handles proposals that contain scanned/signed pages.

Design
------
* Detects scanned pages via a text-density heuristic (characters per pixel area).
* Renders pages to images using PyMuPDF (already a dependency) — no extra install.
* Runs OCR in configurable batches so RAM stays bounded on 200-page documents.
* Caches results in a JSON sidecar file so re-runs are instant.
* Exposes two public functions:

    ocr_proposal_if_needed(pdf_path, force=False) -> dict[int, str]
        Full-document OCR pass; returns {page_1_based: text}.
        Called from ingest_proposal before chunk ingestion.

    get_page_text_ocr(pdf_path, page_numbers_1based, ocr_cache) -> str
        Fast retrieval using an already-built cache; called from
        _get_proposal_pages as a fallback when native text is sparse.

Backend priority
----------------
1. Docling  (already present in environment, best accuracy for complex layouts)
2. pytesseract  (lightweight, works with PyMuPDF-rendered PNGs)
3. EasyOCR  (GPU-optional, good on rotated/handwritten text)

Whichever backend is importable is used; the rest are skipped silently.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pages with fewer than this many characters per 1 000 000 pixels are "scanned"
_TEXT_DENSITY_THRESHOLD = 0.05          # chars / pixel  (empirically tuned)
_MIN_TEXT_LEN           = 80            # also flag if raw text is shorter than this
_DPI                    = 150           # render DPI (150 is fast & good enough for Tesseract)
_BATCH_SIZE             = 20            # pages per OCR batch
_CACHE_SUFFIX           = ".ocr_cache.json"

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _backend() -> str:
    """Return the first available OCR backend name."""
    try:
        import docling  # noqa: F401
        return "docling"
    except ImportError:
        pass
    try:
        import pytesseract  # noqa: F401
        return "pytesseract"
    except ImportError:
        pass
    try:
        import easyocr  # noqa: F401
        return "easyocr"
    except ImportError:
        pass
    return "none"


_BACKEND = _backend()
print(f"[OCR] Backend selected: {_BACKEND}")


# ---------------------------------------------------------------------------
# Page-density scanner
# ---------------------------------------------------------------------------

def _page_is_scanned(page) -> bool:
    """
    Return True when a PyMuPDF page appears to be an image-only (scanned) page.

    Heuristic:
      • If native text length < _MIN_TEXT_LEN → likely scanned
      • If chars / pixel_area < _TEXT_DENSITY_THRESHOLD → likely scanned
    """
    text = page.get_text().strip()
    if len(text) < _MIN_TEXT_LEN:
        return True
    rect        = page.rect
    pixel_area  = rect.width * rect.height
    if pixel_area == 0:
        return False
    density = len(text) / pixel_area
    return density < _TEXT_DENSITY_THRESHOLD


def _find_scanned_pages(pdf_path: str) -> list[int]:
    """Return list of 1-based page numbers that appear to be scanned."""
    import fitz
    doc     = fitz.open(pdf_path)
    scanned = [i + 1 for i in range(len(doc)) if _page_is_scanned(doc[i])]
    doc.close()
    print(f"[OCR] Scanned pages detected ({len(scanned)}/{len(doc) + len(scanned) - len(scanned)}): "
          f"{scanned[:30]}{'…' if len(scanned) > 30 else ''}")
    return scanned


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(pdf_path: str) -> Path:
    return Path(pdf_path).with_suffix(_CACHE_SUFFIX)


def _load_cache(pdf_path: str) -> dict[int, str]:
    cp = _cache_path(pdf_path)
    if cp.exists():
        try:
            raw = json.loads(cp.read_text(encoding="utf-8"))
            # Keys are stored as strings in JSON; convert back to int
            return {int(k): v for k, v in raw.items()}
        except Exception as e:
            print(f"[OCR] Cache read failed ({e}); will rebuild")
    return {}


def _save_cache(pdf_path: str, cache: dict[int, str]) -> None:
    cp = _cache_path(pdf_path)
    try:
        cp.write_text(
            json.dumps({str(k): v for k, v in cache.items()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OCR] Cache saved → {cp}")
    except Exception as e:
        print(f"[OCR] Cache write failed: {e}")


# ---------------------------------------------------------------------------
# Backend: Docling
# ---------------------------------------------------------------------------

def _ocr_batch_docling(pdf_path: str, page_numbers: list[int]) -> dict[int, str]:
    """
    Extract text from specific pages using Docling with OCR enabled.
    Returns {page_1based: text}.
    """
    try:
        import fitz
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    except ImportError as e:
        print(f"[OCR/Docling] Import failed: {e}")
        return {}

    try:
        # Extract only the target pages into a temp PDF
        src = fitz.open(pdf_path)
        dst = fitz.open()
        for pg in page_numbers:
            idx = pg - 1
            if 0 <= idx < len(src):
                dst.insert_pdf(src, from_page=idx, to_page=idx)
        src.close()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            dst.save(f.name)
            tmp_path = f.name
        dst.close()

        opts = PdfPipelineOptions()
        opts.do_ocr             = True
        opts.do_table_structure = False    # not needed for text extraction
        opts.ocr_options.lang   = ["en"]

        conv   = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        result = conv.convert(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        # Docling resets page numbering to 1 for the sub-PDF;
        # map them back to the original page numbers
        texts: dict[int, str] = {}
        for i, orig_pg in enumerate(page_numbers):
            local_pg = i + 1
            # Collect all text items on this local page
            page_text = "\n".join(
                item.text
                for item in result.document.texts
                if getattr(item.prov[0], "page_no", None) == local_pg
                   if getattr(item, "prov", None)
            )
            texts[orig_pg] = page_text.strip()

        return texts

    except Exception as e:
        print(f"[OCR/Docling] Batch failed: {e}")
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return {}


# ---------------------------------------------------------------------------
# Backend: pytesseract
# ---------------------------------------------------------------------------

def _ocr_batch_pytesseract(pdf_path: str, page_numbers: list[int]) -> dict[int, str]:
    """
    Render pages via PyMuPDF to PNG then run Tesseract.
    Returns {page_1based: text}.
    """
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
    except ImportError as e:
        print(f"[OCR/pytesseract] Import failed: {e}")
        return {}

    results: dict[int, str] = {}
    doc = fitz.open(pdf_path)

    for pg in page_numbers:
        idx = pg - 1
        if not (0 <= idx < len(doc)):
            continue
        try:
            mat     = fitz.Matrix(_DPI / 72, _DPI / 72)   # scale to DPI
            pix     = doc[idx].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img     = Image.open(io.BytesIO(pix.tobytes("png")))
            text    = pytesseract.image_to_string(img, lang="eng",
                          config="--oem 3 --psm 6")
            results[pg] = re.sub(r"\n{3,}", "\n\n", text).strip()
        except Exception as e:
            print(f"[OCR/pytesseract] Page {pg} failed: {e}")
            results[pg] = ""

    doc.close()
    return results


# ---------------------------------------------------------------------------
# Backend: EasyOCR
# ---------------------------------------------------------------------------

_EASY_READER = None   # lazy singleton


def _ocr_batch_easyocr(pdf_path: str, page_numbers: list[int]) -> dict[int, str]:
    """
    Render pages via PyMuPDF then run EasyOCR.
    Returns {page_1based: text}.
    """
    global _EASY_READER
    try:
        import fitz
        import easyocr
        import io
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"[OCR/EasyOCR] Import failed: {e}")
        return {}

    if _EASY_READER is None:
        print("[OCR/EasyOCR] Loading model (first use)…")
        _EASY_READER = easyocr.Reader(["en"], gpu=False)

    results: dict[int, str] = {}
    doc = fitz.open(pdf_path)

    for pg in page_numbers:
        idx = pg - 1
        if not (0 <= idx < len(doc)):
            continue
        try:
            mat  = fitz.Matrix(_DPI / 72, _DPI / 72)
            pix  = doc[idx].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            arr  = np.array(img)
            raw  = _EASY_READER.readtext(arr, detail=0, paragraph=True)
            results[pg] = "\n".join(raw).strip()
        except Exception as e:
            print(f"[OCR/EasyOCR] Page {pg} failed: {e}")
            results[pg] = ""

    doc.close()
    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _ocr_batch(pdf_path: str, page_numbers: list[int]) -> dict[int, str]:
    """Run the configured backend on a list of page numbers."""
    if _BACKEND == "docling":
        return _ocr_batch_docling(pdf_path, page_numbers)
    if _BACKEND == "pytesseract":
        return _ocr_batch_pytesseract(pdf_path, page_numbers)
    if _BACKEND == "easyocr":
        return _ocr_batch_easyocr(pdf_path, page_numbers)

    print("[OCR] No backend available — install pytesseract, docling, or easyocr")
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ocr_proposal_if_needed(
    pdf_path: str,
    force: bool = False,
    batch_size: int = _BATCH_SIZE,
) -> dict[int, str]:
    """
    Full-document OCR pass over a proposal PDF.

    Steps:
      1. Load cache; return immediately if complete and not forced.
      2. Identify scanned pages (density heuristic).
      3. OCR in batches of `batch_size` pages.
      4. Save updated cache and return {page_1based: ocr_text}.

    Parameters
    ----------
    pdf_path   : path to the proposal PDF
    force      : re-OCR even if cache exists
    batch_size : pages per batch (lower = less RAM; 20 is a safe default)

    Returns
    -------
    dict mapping 1-based page number → OCR text string
    """
    if _BACKEND == "none":
        print("[OCR] Skipping — no OCR backend installed")
        return {}

    pdf_path = str(pdf_path)
    cache    = {} if force else _load_cache(pdf_path)

    scanned = _find_scanned_pages(pdf_path)
    needed  = [p for p in scanned if p not in cache]

    if not needed:
        print(f"[OCR] All {len(scanned)} scanned pages already cached")
        return cache

    print(f"[OCR] Will OCR {len(needed)} pages in batches of {batch_size}…")
    t0 = time.time()

    for batch_start in range(0, len(needed), batch_size):
        batch = needed[batch_start: batch_start + batch_size]
        print(f"[OCR] Batch {batch_start // batch_size + 1}: "
              f"pages {batch[0]}–{batch[-1]}")
        results = _ocr_batch(pdf_path, batch)
        cache.update(results)
        # Save after each batch so a crash doesn't lose everything
        _save_cache(pdf_path, cache)

    elapsed = time.time() - t0
    print(f"[OCR] Done — {len(needed)} pages OCR'd in {elapsed:.1f}s "
          f"({elapsed / max(len(needed), 1):.1f}s/page)")
    return cache


def get_page_text_ocr(
    pdf_path: str,
    page_numbers_1based: list[int],
    ocr_cache: Optional[dict[int, str]] = None,
    max_chars: int = 40_000,
) -> str:
    """
    Return text for the given pages, using OCR cache where available and
    falling back to on-demand OCR for any uncached pages.

    This is a drop-in supplement for _get_proposal_pages when native
    PyMuPDF text is sparse.

    Parameters
    ----------
    pdf_path            : path to proposal PDF
    page_numbers_1based : pages to retrieve
    ocr_cache           : previously built cache (pass the result of
                          ocr_proposal_if_needed to avoid re-loading JSON)
    max_chars           : character cap on returned text

    Returns
    -------
    str  — concatenated text from the requested pages
    """
    if ocr_cache is None:
        ocr_cache = _load_cache(str(pdf_path))

    missing = [p for p in page_numbers_1based if p not in ocr_cache]
    if missing:
        print(f"[OCR] On-demand OCR for {len(missing)} uncached pages: {missing}")
        new_results = _ocr_batch(str(pdf_path), missing)
        ocr_cache.update(new_results)
        if new_results:
            _save_cache(str(pdf_path), ocr_cache)

    parts = []
    total = 0
    for pg in page_numbers_1based:
        text = ocr_cache.get(pg, "")
        if not text:
            continue
        block  = f"[Page {pg}]\n{text}"
        if total + len(block) > max_chars:
            block = block[: max_chars - total]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break

    return "\n\n---\n\n".join(parts)


def augment_chunks_with_ocr(
    chunks: list,
    pdf_path: str,
    ocr_cache: Optional[dict[int, str]] = None,
) -> list:
    """
    Walk through parsed chunks and replace empty/thin text with OCR text.

    This is called from ingest_proposal after parse_document returns chunks.
    Chunks whose text is blank or very short (< _MIN_TEXT_LEN chars) get their
    text replaced by the OCR result for the same page.

    Parameters
    ----------
    chunks    : list of chunk objects from core.parser.parse_document
    pdf_path  : path to proposal PDF (used to find the OCR cache)
    ocr_cache : pre-built cache from ocr_proposal_if_needed (optional)

    Returns
    -------
    The same list with text fields updated in-place.
    """
    if ocr_cache is None:
        ocr_cache = _load_cache(str(pdf_path))

    enriched = 0
    for chunk in chunks:
        page  = getattr(chunk, "page_no", 0)
        ctext = getattr(chunk, "text", "") or ""
        if len(ctext.strip()) < _MIN_TEXT_LEN and page in ocr_cache:
            ocr_text = ocr_cache[page].strip()
            if len(ocr_text) > len(ctext.strip()):
                chunk.text = ocr_text
                enriched  += 1

    print(f"[OCR] augment_chunks_with_ocr: enriched {enriched}/{len(chunks)} chunks")
    return chunks
