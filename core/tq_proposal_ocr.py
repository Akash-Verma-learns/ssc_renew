"""
tq_proposal_ocr.py  (v5 — Ollama-powered OCR correction + smart page classification)
======================================================================================

v4 → v5 changes
----------------

NEW 1 — LLM OCR Correction (_llm_correct_ocr_text):
  After raw OCR, pages that contain numbers, table structures, or financial data
  are passed through Ollama for correction of common OCR errors:
  - "1NR" → "INR", "Crore5" → "Crores", "rnarks" → "marks"
  - Garbled table rows reconstructed from context
  - Confidence: only applies corrections when Ollama is certain

NEW 2 — LLM Page Content Classification (_llm_classify_page_content):
  Ollama classifies each OCR'd page into one of:
  - COMPLIANCE_TABLE: vendor's own compliance/response table
  - SCORING_TABLE: RFP's evaluation criteria table
  - PROJECT_LIST: vendor's project/assignment listing
  - FINANCIAL_DATA: CA certificates, balance sheet data
  - TEAM_CV: team member CVs and qualifications
  - NARRATIVE: general narrative/methodology text
  - BOILERPLATE: headers, footers, blank pages
  This classification is stored in the cache and used by tq_extractor.py
  to route page searches more intelligently.

NEW 3 — Smart Compliance Table Detection (_llm_detect_compliance_start):
  Instead of regex-only, uses Ollama to reason about which page contains
  the vendor's compliance table, even when headers are non-standard.

NEW 4 — Structured Data Extraction from OCR'd Tables (_extract_table_data_from_page):
  For FINANCIAL_DATA and COMPLIANCE_TABLE pages, Ollama extracts structured
  key-value pairs (turnover figures, project counts, years) directly,
  pre-computing values that the extractor can use without re-reading.

FIX 4 retained: PIL Image → numpy before ocr.ocr() call.
All v4 functionality retained.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DPI            = 200
DPI_TESSERACT  = 300
TEXT_THRESHOLD = 150
MAX_WORKERS    = 3
CACHE_SUFFIX   = "_ocr_cache.json"

OLLAMA_MODEL     = "llama3.2"
OLLAMA_HOST      = "http://localhost:11434"
OLLAMA_CHAT_URL  = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TIMEOUT_S = 60    # per-page correction: short timeout
OLLAMA_TIMEOUT_C = 30    # classification: very short

_HEADER_PATTERNS = [
    "proposal for providing",
    "request for proposal",
    "grant thornton",
    "confidential",
    "p a g e",
]

# Page content types
PAGE_TYPE_COMPLIANCE  = "COMPLIANCE_TABLE"
PAGE_TYPE_SCORING     = "SCORING_TABLE"
PAGE_TYPE_PROJECT     = "PROJECT_LIST"
PAGE_TYPE_FINANCIAL   = "FINANCIAL_DATA"
PAGE_TYPE_TEAM_CV     = "TEAM_CV"
PAGE_TYPE_NARRATIVE   = "NARRATIVE"
PAGE_TYPE_BOILERPLATE = "BOILERPLATE"


# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, ctx: int = 4096,
                 timeout: int = OLLAMA_TIMEOUT_S) -> str:
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={"model": OLLAMA_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {"temperature": 0.0, "num_ctx": ctx}},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"] or ""
    except requests.exceptions.Timeout:
        return ""
    except Exception as e:
        print(f"[OCR] Ollama error: {e}"); return ""


def _parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    start = text.find("{")
    if start < 0: return None
    text = text[start:]
    end  = text.rfind("}") + 1
    candidate = text[:end] if end > 0 else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# NEW v5: LLM OCR Correction
# ---------------------------------------------------------------------------

_OCR_CORRECTION_PROMPT = """\
You are correcting OCR errors in a page from an Indian government consulting proposal.

Common OCR errors in these documents:
- "1NR" or "lNR" → "INR" (Indian Rupees prefix)
- "Crore5" or "Crore$" → "Crores"
- "rnarks" or "rnax" → "marks" or "max"
- "rnillion" → "million"
- "0" (zero) confused with "O" (letter) in numbers
- Broken table rows where cells run together
- "5 rnark5" → "5 marks"
- Numbers split by line breaks: "12.\n30" → "12.30"
- "lNR 12.30 Crore5" → "INR 12.30 Crores"

RULES:
1. Fix ONLY clear OCR errors. Do NOT rephrase or add information.
2. Preserve all numbers exactly as they appear (after correcting obvious errors).
3. Preserve table structure (pipes, dashes, spaces).
4. If a line looks like "2018-19 13.55" it is a financial year + turnover figure.
5. Return the corrected text only, with no explanation.

ORIGINAL OCR TEXT:
{ocr_text}

Return ONLY the corrected text:"""

_OCR_NEEDS_CORRECTION_SIGNALS = re.compile(
    r'(1NR|lNR|\brnark|\brnaxi|\bCrore[^s]|0rder|[0-9][Oo][0-9]|'
    r'\b[0-9]+\s*\.\s*[0-9]+\s*Cr|\bINR\s+[0-9])',
    re.IGNORECASE
)

_FINANCIAL_PAGE_SIGNALS = re.compile(
    r'(turnover|crore|inr|rs\.?|balance\s+sheet|audited|financial\s+year|'
    r'ca\s+certificate|advisory|revenue)',
    re.IGNORECASE
)

_TABLE_PAGE_SIGNALS = re.compile(
    r'(marks?|evaluation|criteria|s\.?\s*no|parameter|max|points?)',
    re.IGNORECASE
)


def _should_correct_ocr(page_text: str) -> bool:
    """Decide if a page needs LLM OCR correction."""
    if len(page_text) < 50:
        return False
    has_financial  = bool(_FINANCIAL_PAGE_SIGNALS.search(page_text))
    has_table      = bool(_TABLE_PAGE_SIGNALS.search(page_text))
    has_errors     = bool(_OCR_NEEDS_CORRECTION_SIGNALS.search(page_text))
    return (has_financial or has_table) and (has_errors or len(page_text) > 300)


def _llm_correct_ocr_text(page_text: str, page_no: int) -> str:
    """
    Use Ollama to correct OCR errors on a page.
    Only called for pages with financial or table content.
    Returns corrected text, or original if LLM fails/times out.
    """
    if not _should_correct_ocr(page_text):
        return page_text

    prompt = _OCR_CORRECTION_PROMPT.format(
        ocr_text=page_text[:3000])  # limit to avoid context overflow

    corrected = _call_ollama(prompt, ctx=4096, timeout=OLLAMA_TIMEOUT_S)

    if not corrected or len(corrected) < len(page_text) * 0.5:
        # LLM returned nothing useful, keep original
        return page_text

    # Sanity check: corrected text should have similar length
    if len(corrected) > len(page_text) * 2:
        print(f"[OCR] LLM correction p{page_no}: result too long, keeping original")
        return page_text

    print(f"[OCR] LLM correction p{page_no}: "
          f"{len(page_text)} → {len(corrected)} chars")
    return corrected


# ---------------------------------------------------------------------------
# NEW v5: LLM Page Classification
# ---------------------------------------------------------------------------

_PAGE_CLASSIFICATION_PROMPT = """\
You are analyzing a page from an Indian government consulting proposal.

Classify this page into EXACTLY ONE of these categories:

COMPLIANCE_TABLE: The vendor's own compliance/response table showing how they meet
  each RFP evaluation criterion. Contains phrases like "our compliance",
  "GT Marks", "Reference Document", "GT has experience", compliance responses.

SCORING_TABLE: The RFP's own evaluation criteria table with S.No, Parameters,
  Max Marks columns. Contains the scoring rules from the RFP itself.

PROJECT_LIST: A table or list of the vendor's past projects/assignments with
  client names, contract values, dates, descriptions.

FINANCIAL_DATA: CA certificates, audited balance sheets, turnover certificates,
  financial year data with revenue figures.

TEAM_CV: Curriculum vitae of proposed team members, team composition tables,
  Form Tech-4/5 type content.

NARRATIVE: Approach, methodology, work plan, technical narrative, company profile.

BOILERPLATE: Cover page, table of contents, section dividers, nearly blank pages,
  disclaimer pages, form templates without content.

PAGE TEXT:
{page_text}

Return ONLY valid JSON:
{{"page_type": "<one of the categories above>", "confidence": <0.0-1.0>,
  "key_signals": "<2-3 words that led to classification>"}}
"""

_FINANCIAL_EXTRACTION_PROMPT = """\
You are extracting financial figures from a page of an Indian government proposal.

This page contains financial/turnover data. Extract ALL financial figures.

Common formats:
- "Average Annual Turnover of INR 12.30 Crores"
- Table: "2018-19 | 13.55 | 2017-18 | 13.19 | 2016-17 | 10.15 | Average | 12.30"
- "Total 36.89, Average Annual Turnover 36.89/3 = 12.30"
- "INR 230.00 crore from overall advisory services"

PAGE TEXT:
{page_text}

Return ONLY valid JSON:
{{
  "figures": [
    {{
      "description": "<what this figure is>",
      "value_crore": <float — value in crores>,
      "period": "<financial years e.g. 2018-19 to 2016-17 or 'average'>",
      "type": "<agri_advisory / overall_advisory / project_fee / other>"
    }}
  ]
}}
"""

_PROJECT_EXTRACTION_PROMPT = """\
You are extracting project/assignment information from a page of an Indian government proposal.

Extract ALL projects/assignments mentioned. Focus on:
- Assignment name
- Client name and type (Central Govt / State Govt / Multilateral / Private)
- Duration (months or years)
- Professional fees (in INR crores)
- Year of contract
- Whether it's in horticulture/agri sector

PAGE TEXT:
{page_text}

Return ONLY valid JSON:
{{
  "projects": [
    {{
      "name": "<assignment name>",
      "client": "<client name>",
      "client_type": "<Central Govt / State Govt / Multilateral / Private>",
      "duration_months": <int or null>,
      "fee_crore": <float or null>,
      "year": <int or null>,
      "sector": "<horticulture / agriculture / rural / urban / other>",
      "is_pma_pmu": <true/false>
    }}
  ]
}}
"""


def _llm_classify_page(page_text: str, page_no: int) -> dict:
    """
    Use Ollama to classify a page's content type.
    Returns dict with page_type, confidence, key_signals.
    Falls back to rule-based classification if LLM unavailable.
    """
    if len(page_text.strip()) < 30:
        return {"page_type": PAGE_TYPE_BOILERPLATE,
                "confidence": 1.0, "key_signals": "blank"}

    # Fast rule-based pre-classification to avoid LLM for obvious cases
    lower = page_text.lower()
    if any(p in lower for p in ["table of contents", "disclaimer", "section -",
                                  "letter of invitation"]):
        return {"page_type": PAGE_TYPE_BOILERPLATE,
                "confidence": 0.9, "key_signals": "toc/disclaimer"}

    prompt = _PAGE_CLASSIFICATION_PROMPT.format(
        page_text=page_text[:2000])
    raw = _call_ollama(prompt, ctx=2048, timeout=OLLAMA_TIMEOUT_C)
    if raw:
        data = _parse_json(raw)
        if data and data.get("page_type"):
            return data

    # Rule-based fallback
    if re.search(r'our\s+compliance|gt\s+marks|reference\s+document', lower):
        return {"page_type": PAGE_TYPE_COMPLIANCE,
                "confidence": 0.8, "key_signals": "our compliance"}
    if re.search(r's\.?\s*no|max\s+marks|evaluation\s+criteria', lower):
        return {"page_type": PAGE_TYPE_SCORING,
                "confidence": 0.7, "key_signals": "marks table"}
    if re.search(r'turnover|ca\s+certificate|balance\s+sheet', lower):
        return {"page_type": PAGE_TYPE_FINANCIAL,
                "confidence": 0.7, "key_signals": "financial"}
    if re.search(r'curriculum\s+vitae|form\s+tech.?5|proposed\s+position', lower):
        return {"page_type": PAGE_TYPE_TEAM_CV,
                "confidence": 0.7, "key_signals": "cv"}

    return {"page_type": PAGE_TYPE_NARRATIVE,
            "confidence": 0.5, "key_signals": "default"}


def _llm_extract_page_data(page_text: str, page_type: str,
                            page_no: int) -> Optional[dict]:
    """
    For high-value pages (FINANCIAL_DATA, PROJECT_LIST, COMPLIANCE_TABLE),
    extract structured data using Ollama.
    Returns structured dict or None.
    """
    if page_type == PAGE_TYPE_FINANCIAL:
        prompt = _FINANCIAL_EXTRACTION_PROMPT.format(
            page_text=page_text[:3000])
        raw = _call_ollama(prompt, ctx=4096, timeout=OLLAMA_TIMEOUT_S)
        if raw:
            data = _parse_json(raw)
            if data and data.get("figures"):
                print(f"[OCR] p{page_no} FINANCIAL: "
                      f"extracted {len(data['figures'])} figures")
                return {"type": "financial", "data": data["figures"]}

    elif page_type == PAGE_TYPE_PROJECT:
        prompt = _PROJECT_EXTRACTION_PROMPT.format(
            page_text=page_text[:3000])
        raw = _call_ollama(prompt, ctx=4096, timeout=OLLAMA_TIMEOUT_S)
        if raw:
            data = _parse_json(raw)
            if data and data.get("projects"):
                print(f"[OCR] p{page_no} PROJECTS: "
                      f"extracted {len(data['projects'])} projects")
                return {"type": "projects", "data": data["projects"]}

    return None


# ---------------------------------------------------------------------------
# NEW v5: LLM Compliance Table Detection
# ---------------------------------------------------------------------------

_COMPLIANCE_DETECT_PROMPT = """\
You are analyzing pages from an Indian government consulting proposal (technical bid).

I need to find which page contains the vendor's OWN COMPLIANCE TABLE — the table
where the vendor maps their experience against the RFP's evaluation criteria.

Characteristics of a compliance table:
- Has columns like: "S.No | Criteria | Proposed Marks | Vendor Marks | Reference Document"
- Contains vendor responses like "GT has experience of...", "Please refer Form Tech-2..."
- Shows how vendor scores against each criterion
- Usually titled "Technical Evaluation Criteria and Our Compliance" or similar
- Different from the RFP's OWN scoring table (which has blank marks columns)

PAGE SUMMARIES (first 200 chars each):
{page_summaries}

Return ONLY valid JSON:
{{
  "compliance_table_page": <page number or null if not found>,
  "confidence": <0.0-1.0>,
  "evidence": "<what you saw on that page>"
}}
"""


def _llm_detect_compliance_start(page_texts: dict) -> Optional[int]:
    """
    Use Ollama to find the compliance table start page.
    More robust than regex for non-standard headers.
    """
    # First try fast regex
    _SIGNALS = re.compile(
        r'(our\s+compliance|'
        r'technical\s+bid\s+evaluation\s+criteria\s+and\s+our|'
        r'compliance\s+(?:statement|table|against)|'
        r'page\s+nos?\.?\s+for\s+supporting\s+documents|'
        r'gt\s+marks|proposed\s+marks)',
        re.IGNORECASE
    )
    for pno in sorted(page_texts.keys()):
        if _SIGNALS.search(page_texts.get(pno, "")):
            print(f"[OCR] Compliance table (regex): page {pno}")
            return pno

    # LLM fallback — check a range of pages
    pages_to_check = sorted(page_texts.keys())
    # Focus on middle-to-end pages (compliance table is usually after methodology)
    if len(pages_to_check) > 20:
        pages_to_check = pages_to_check[len(pages_to_check)//3:]

    # Build summaries of candidate pages
    summaries = []
    for pno in pages_to_check[:30]:  # max 30 pages to LLM
        txt = page_texts.get(pno, "")
        if txt and len(txt.strip()) > 50:
            summaries.append(f"PAGE {pno}: {txt[:200]}")

    if not summaries:
        return None

    prompt = _COMPLIANCE_DETECT_PROMPT.format(
        page_summaries="\n\n".join(summaries))
    raw = _call_ollama(prompt, ctx=8192, timeout=OLLAMA_TIMEOUT_S)
    if raw:
        data = _parse_json(raw)
        if data and data.get("compliance_table_page"):
            pg = int(data["compliance_table_page"])
            conf = float(data.get("confidence", 0))
            if conf >= 0.5:
                print(f"[OCR] Compliance table (LLM): page {pg} "
                      f"(confidence={conf:.2f}) — {data.get('evidence','')[:60]}")
                return pg

    return None


# ---------------------------------------------------------------------------
# Backend detection (unchanged from v4)
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    try:
        import paddleocr
        import paddle
        return "paddle"
    except ImportError:
        pass
    tess = _find_tesseract()
    if tess:
        return "tesseract"
    return "none"


def _find_tesseract() -> Optional[str]:
    found = shutil.which("tesseract")
    if found:
        return found
    if platform.system() == "Windows":
        for p in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\akash\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        ]:
            if Path(p).exists():
                return p
    return None


_BACKEND: str                 = _detect_backend()
_TESSERACT_CMD: Optional[str] = (_find_tesseract()
                                  if _BACKEND == "tesseract" else None)

if _BACKEND == "tesseract" and _TESSERACT_CMD:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    except ImportError:
        pass

print(f"[OCR] Backend: {_BACKEND}" +
      (f" @ {_TESSERACT_CMD}" if _TESSERACT_CMD else ""))


# ---------------------------------------------------------------------------
# PaddleOCR singleton
# ---------------------------------------------------------------------------

_paddle_ocr = None


def _get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        import os
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang="en",
        )
    return _paddle_ocr


def _paddle_ocr_image(img) -> str:
    """
    Run PaddleOCR on a PIL Image.
    FIX (v4): Always convert PIL → numpy first.
    """
    import numpy as np
    try:
        ocr = _get_paddle_ocr()
        img_array = np.array(img)
        results = ocr.ocr(img_array)
        if not results:
            return ""
        lines = []
        for page_result in results:
            if not page_result:
                continue
            if isinstance(page_result, list):
                for item in page_result:
                    try:
                        text, conf = item[1][0], item[1][1]
                        if conf >= 0.4:
                            lines.append(text)
                    except Exception:
                        pass
            elif hasattr(page_result, "boxes"):
                for box in page_result.boxes:
                    if box.score >= 0.4:
                        lines.append(box.rec_text)
        return "\n".join(lines)

    except Exception as e:
        print(f"[OCR] Paddle failed → fallback to Tesseract: {e}")
        try:
            import pytesseract
            from PIL import ImageEnhance, ImageFilter
            if _TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
            img_gray = img.convert("L")
            img_gray = ImageEnhance.Contrast(img_gray).enhance(1.5)
            img_gray = img_gray.filter(ImageFilter.SHARPEN)
            return pytesseract.image_to_string(
                img_gray, lang="eng", config="--oem 1 --psm 6")
        except Exception as e2:
            print(f"[OCR] Fallback also failed: {e2}")
            return ""


# ---------------------------------------------------------------------------
# Utility helpers (unchanged from v4)
# ---------------------------------------------------------------------------

def _is_header_line(line: str) -> bool:
    low = line.lower().strip()
    return not low or low.isdigit() or any(p in low for p in _HEADER_PATTERNS)


def _real_text_length(raw: str) -> int:
    return sum(len(l.strip()) for l in raw.split("\n")
               if not _is_header_line(l))


def _cache_path(proposal_path: str) -> Path:
    p = Path(proposal_path)
    return p.parent / (p.stem + CACHE_SUFFIX)


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Single-page render + OCR (v5: adds LLM correction)
# ---------------------------------------------------------------------------

def _ocr_single_page(proposal_path: str, pno_0: int,
                      apply_llm_correction: bool = True) -> tuple[int, str]:
    pno_1 = pno_0 + 1
    try:
        import fitz, io
        from PIL import Image

        doc  = fitz.open(proposal_path)
        page = doc[pno_0]

        if _BACKEND == "paddle":
            pix = page.get_pixmap(dpi=DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            doc.close()
            text = _paddle_ocr_image(img)

        elif _BACKEND == "tesseract":
            import pytesseract
            from PIL import ImageEnhance, ImageFilter
            if _TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
            pix = page.get_pixmap(dpi=DPI_TESSERACT)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
            img = ImageEnhance.Contrast(img).enhance(1.5)
            img = img.filter(ImageFilter.SHARPEN)
            doc.close()
            text = pytesseract.image_to_string(
                img, lang="eng", config="--oem 1 --psm 6")
        else:
            doc.close()
            text = ""

        text = text.strip()

        # v5: Apply LLM OCR correction for financial/table pages
        if apply_llm_correction and text:
            text = _llm_correct_ocr_text(text, pno_1)

        return pno_1, text

    except Exception as e:
        return pno_1, f"[OCR ERROR p{pno_1}: {e}]"


def _tesseract_worker(args: tuple) -> tuple[int, str]:
    proposal_path, pno_0, tesseract_cmd = args
    pno_1 = pno_0 + 1
    try:
        import fitz, io, pytesseract
        from PIL import Image, ImageEnhance, ImageFilter
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        doc  = fitz.open(proposal_path)
        pix  = doc[pno_0].get_pixmap(dpi=DPI_TESSERACT)
        doc.close()
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = img.filter(ImageFilter.SHARPEN)
        text = pytesseract.image_to_string(
            img, lang="eng", config="--oem 1 --psm 6")
        return pno_1, text.strip()
    except Exception as e:
        return pno_1, f"[OCR ERROR p{pno_1}: {e}]"


# ---------------------------------------------------------------------------
# UPGRADED Main OCR function (v5: classification + structured extraction)
# ---------------------------------------------------------------------------

def get_proposal_text(
    proposal_path: str,
    pages: Optional[list] = None,
    force_refresh: bool = False,
    progress_callback=None,
    classify_pages: bool = True,
    extract_structured: bool = True,
) -> dict:
    """
    OCR all pages of a proposal and return a dict of {page_no: text}.

    v5 additions:
    - classify_pages: run LLM page classification on high-value pages
    - extract_structured: run LLM structured data extraction on
      FINANCIAL_DATA and PROJECT_LIST pages
    Both results are stored in the cache for use by tq_extractor.py.
    """
    path       = str(proposal_path)
    cache_file = _cache_path(path)
    file_hash  = _file_hash(path)

    cached: dict = {}
    page_types: dict = {}
    structured_data: dict = {}

    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("file_hash") == file_hash:
                cached       = {int(k): v
                                for k, v in data.get("pages", {}).items()}
                page_types   = {int(k): v
                                for k, v in data.get("page_types", {}).items()}
                structured_data = {int(k): v
                                   for k, v in data.get("structured_data",
                                                        {}).items()}
                print(f"[OCR] Cache: {len(cached)} pages from {cache_file.name}")
            else:
                print("[OCR] Cache stale — re-OCR required")
        except Exception as e:
            print(f"[OCR] Cache read error: {e}")

    try:
        import fitz
        doc     = fitz.open(path)
        n_pages = len(doc)
    except Exception as e:
        print(f"[OCR] Cannot open: {e}")
        return {}

    target = set(pages) if pages else set(range(1, n_pages + 1))

    text_native: dict = {}
    needs_ocr: list   = []

    for pno_1 in sorted(target):
        pno_0 = pno_1 - 1
        if pno_0 >= n_pages:
            continue
        if pno_1 in cached:
            text_native[pno_1] = cached[pno_1]
            continue
        raw = doc[pno_0].get_text()
        if _real_text_length(raw) >= TEXT_THRESHOLD:
            text_native[pno_1] = raw.strip()
        else:
            needs_ocr.append(pno_1)

    doc.close()
    print(f"[OCR] {len(target)} pages: {len(text_native)} native, "
          f"{len(needs_ocr)} need OCR [{_BACKEND}]")

    ocr_results: dict = {}

    if needs_ocr and _BACKEND == "none":
        print("[OCR] No OCR backend available.")
        print("[OCR] Install PaddleOCR:  pip install paddleocr paddlepaddle")
        print("[OCR] Or Tesseract:       pip install pytesseract Pillow")
        for pno in needs_ocr:
            ocr_results[pno] = ""

    elif needs_ocr and _BACKEND == "paddle":
        print(f"[OCR] PaddleOCR: {len(needs_ocr)} pages...")
        t0 = time.time(); done = 0
        for pno_1 in needs_ocr:
            pno_1_out, txt = _ocr_single_page(path, pno_1 - 1,
                                               apply_llm_correction=True)
            ocr_results[pno_1_out] = txt
            done += 1
            if progress_callback:
                try: progress_callback(done, len(needs_ocr), pno_1)
                except Exception: pass
            if done % 10 == 0 or done == len(needs_ocr):
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed else 1
                print(f"[OCR] {done}/{len(needs_ocr)} | "
                      f"{elapsed:.0f}s | ETA {(len(needs_ocr)-done)/rate:.0f}s")
        print(f"[OCR] PaddleOCR done in {time.time()-t0:.1f}s")

    elif needs_ocr and _BACKEND == "tesseract":
        print(f"[OCR] Tesseract: {len(needs_ocr)} pages...")
        t0    = time.time(); done = 0
        wargs = [(path, pno - 1, _TESSERACT_CMD) for pno in needs_ocr]
        with ProcessPoolExecutor(
                max_workers=min(MAX_WORKERS, len(needs_ocr))) as pool:
            futs = {pool.submit(_tesseract_worker, a): a for a in wargs}
            for fut in as_completed(futs):
                pno_1, txt = fut.result()
                # Apply LLM correction after parallel OCR
                if txt:
                    txt = _llm_correct_ocr_text(txt, pno_1)
                ocr_results[pno_1] = txt
                done += 1
                if progress_callback:
                    try: progress_callback(done, len(needs_ocr), pno_1)
                    except Exception: pass
                if done % 20 == 0 or done == len(needs_ocr):
                    elapsed = time.time() - t0
                    rate    = done / elapsed if elapsed else 1
                    print(f"[OCR] {done}/{len(needs_ocr)} | "
                          f"{elapsed:.0f}s | "
                          f"ETA {(len(needs_ocr)-done)/rate:.0f}s")
        print(f"[OCR] Tesseract done in {time.time()-t0:.1f}s")

    all_results = {**text_native, **ocr_results}

    # ── v5: LLM Page Classification ────────────────────────────────────────
    if classify_pages:
        print(f"[OCR] Classifying {len(all_results)} pages...")
        t0 = time.time()
        new_classifications = 0
        for pno_1, txt in sorted(all_results.items()):
            if pno_1 in page_types:
                continue  # already classified in cache
            if not txt or len(txt.strip()) < 30:
                page_types[pno_1] = {"page_type": PAGE_TYPE_BOILERPLATE,
                                      "confidence": 1.0}
                continue
            classification = _llm_classify_page(txt, pno_1)
            page_types[pno_1] = classification
            new_classifications += 1
            if new_classifications % 10 == 0:
                print(f"[OCR] Classified {new_classifications} pages...")
        print(f"[OCR] Classification done in {time.time()-t0:.1f}s: "
              f"{new_classifications} new")

        # Print page type summary
        type_counts: dict = {}
        for v in page_types.values():
            pt = v.get("page_type", "?")
            type_counts[pt] = type_counts.get(pt, 0) + 1
        print(f"[OCR] Page types: {type_counts}")

    # ── v5: Structured Data Extraction ─────────────────────────────────────
    if extract_structured and classify_pages:
        high_value_types = {PAGE_TYPE_FINANCIAL, PAGE_TYPE_PROJECT,
                             PAGE_TYPE_COMPLIANCE}
        for pno_1, classification in page_types.items():
            if pno_1 in structured_data:
                continue  # already extracted
            pt = classification.get("page_type", "")
            if pt not in high_value_types:
                continue
            txt = all_results.get(pno_1, "")
            if not txt or len(txt.strip()) < 50:
                continue
            data = _llm_extract_page_data(txt, pt, pno_1)
            if data:
                structured_data[pno_1] = data

        if structured_data:
            print(f"[OCR] Structured data extracted from "
                  f"{len(structured_data)} pages")

    # Update cache
    updated = {int(k): v for k, v in cached.items()}
    updated.update(all_results)

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "file_hash":       file_hash,
                "proposal":        Path(path).name,
                "backend":         _BACKEND,
                "dpi":             DPI if _BACKEND == "paddle" else DPI_TESSERACT,
                "pages":           {str(k): v
                                    for k, v in sorted(updated.items())},
                "page_types":      {str(k): v
                                    for k, v in sorted(page_types.items())},
                "structured_data": {str(k): v
                                    for k, v in sorted(structured_data.items())},
            }, f, ensure_ascii=False)
        print(f"[OCR] Cache saved: {cache_file.name} ({len(updated)} pages)")
    except Exception as e:
        print(f"[OCR] Cache write failed (non-fatal): {e}")

    return all_results


# ---------------------------------------------------------------------------
# Cache loading (v5: also returns page_types and structured_data)
# ---------------------------------------------------------------------------

def load_ocr_cache(proposal_path: str) -> dict:
    """Load cached OCR text. Returns {page_no: text}."""
    cache_file = _cache_path(str(proposal_path))
    if not cache_file.exists():
        return {}
    try:
        file_hash = _file_hash(str(proposal_path))
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("file_hash") != file_hash:
            return {}
        return {int(k): v for k, v in data.get("pages", {}).items()}
    except Exception as e:
        print(f"[OCR] load_ocr_cache error: {e}")
        return {}


def load_page_types(proposal_path: str) -> dict:
    """Load cached page type classifications. Returns {page_no: classification_dict}."""
    cache_file = _cache_path(str(proposal_path))
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("page_types", {}).items()}
    except Exception:
        return {}


def load_structured_data(proposal_path: str) -> dict:
    """Load cached structured data extractions. Returns {page_no: data_dict}."""
    cache_file = _cache_path(str(proposal_path))
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.get("structured_data", {}).items()}
    except Exception:
        return {}


def get_pages_by_type(proposal_path: str, page_type: str) -> list[int]:
    """
    Return page numbers classified as a specific type.
    Uses cached classifications.
    """
    page_types = load_page_types(proposal_path)
    return sorted(
        pno for pno, cls in page_types.items()
        if cls.get("page_type") == page_type
        and cls.get("confidence", 0) >= 0.5
    )


def get_financial_figures(proposal_path: str) -> list[dict]:
    """
    Return all pre-extracted financial figures from the proposal.
    Each entry: {description, value_crore, period, type, page_no}
    """
    structured = load_structured_data(proposal_path)
    figures = []
    for pno, data in sorted(structured.items()):
        if data.get("type") == "financial":
            for fig in data.get("data", []):
                figures.append({**fig, "page_no": pno})
    return figures


def get_project_list(proposal_path: str) -> list[dict]:
    """
    Return all pre-extracted project records from the proposal.
    """
    structured = load_structured_data(proposal_path)
    projects = []
    for pno, data in sorted(structured.items()):
        if data.get("type") == "projects":
            for proj in data.get("data", []):
                projects.append({**proj, "page_no": pno})
    return projects


# ---------------------------------------------------------------------------
# Keyword-scored page retrieval (unchanged from v4, used as fallback)
# ---------------------------------------------------------------------------

def get_pages_by_keyword(
    proposal_path: str,
    keywords: list,
    max_pages: int = 5,
    max_chars: int = 4_000,
    pre_cached: Optional[dict] = None,
) -> str:
    page_texts: dict = {}
    if pre_cached is not None:
        page_texts = pre_cached
    else:
        cache_file = _cache_path(proposal_path)
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("file_hash") == _file_hash(proposal_path):
                    page_texts = {int(k): v
                                  for k, v in data.get("pages", {}).items()}
            except Exception:
                pass

    if not page_texts:
        try:
            import fitz
            doc = fitz.open(proposal_path)
            page_texts = {pno + 1: doc[pno].get_text().strip()
                          for pno in range(len(doc))}
            doc.close()
        except Exception:
            return ""

    kws    = [k.lower() for k in keywords]
    scored = [(sum(txt.lower().count(kw) for kw in kws), pno, txt)
              for pno, txt in page_texts.items()
              if sum(txt.lower().count(kw) for kw in kws) > 0]
    scored.sort(reverse=True)

    parts, total = [], 0
    for _, pno, txt in scored[:max_pages]:
        block = f"\n[Page {pno}]\n{txt}"
        if total + len(block) > max_chars:
            block = block[:max_chars - total]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break

    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Background OCR entry point (v5: includes classification)
# ---------------------------------------------------------------------------

def run_background_ocr(proposal_path: str,
                        progress_callback=None) -> dict:
    print(f"[OCR] Starting [{_BACKEND}]: {Path(proposal_path).name}")
    results = get_proposal_text(
        proposal_path,
        force_refresh=False,
        progress_callback=progress_callback,
        classify_pages=True,
        extract_structured=True,
    )
    print(f"[OCR] Done: {len(results)} pages")

    # Report high-value page counts
    page_types = load_page_types(proposal_path)
    if page_types:
        compliance_pages  = [p for p, t in page_types.items()
                              if t.get("page_type") == PAGE_TYPE_COMPLIANCE]
        financial_pages   = [p for p, t in page_types.items()
                              if t.get("page_type") == PAGE_TYPE_FINANCIAL]
        project_pages     = [p for p, t in page_types.items()
                              if t.get("page_type") == PAGE_TYPE_PROJECT]
        print(f"[OCR] Key pages — compliance:{compliance_pages} "
              f"financial:{financial_pages} projects:{project_pages}")

    return results