"""
tq_step1_extract.py  —  Deterministic PyMuPDF marking-scheme extractor
=======================================================================

Zero LLM.  All extraction is geometry + regex on the actual PDF bytes.

Pipeline
--------
1. _find_eval_pages()     : scan TOC + body to get start/end page numbers.
2. _collect_words()       : pull word-level bounding boxes from those pages.
3. _calibrate_columns()   : find column X positions from the table HEADER row.
4. _find_sno_anchors()    : locate row numbers 1-N using calibrated S.No X.
5. _extract_rows()        : for each anchor pull parameter + criteria + marks.
6. Fallback               : text-line reconstruction when geometry fails.
7. Validate               : drop presentation/financial/sub-item rows.

Key fix vs previous version
----------------------------
BUG: left_limit = page_width * 0.11 = 67px.
     Standard A4 table left margin ≈ 72pt (1 inch).
     All S.No anchors were outside the search window → 0 anchors found.

FIX: Calibrate the S.No column X from the actual "S. No." header words.
     If that fails, use page_width * 0.20 (= 122px) as the fallback limit.
     Also added cross-validation: accept a digit as S.No only if a
     corresponding marks-range integer exists in the same row region.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    import fitz           # PyMuPDF
except ImportError:
    raise ImportError("pip install pymupdf")

UPLOAD_DIR = Path("./uploads")

# ── Skip patterns ────────────────────────────────────────────────────────────
_SKIP_PARAMS = re.compile(
    r"(presentation|interview|viva|demo|panel|financial\s+bid|price\s+bid"
    r"|quoted\s+rate|\bL1\b|commercial\s+bid|indemnity|arbitration"
    r"|commencement|award\s+of|penalty|force\s+majeure)",
    re.IGNORECASE,
)

# Sub-item expert roles that appear INSIDE the Qualifications cell
_SUBITEM_ROLE = re.compile(
    r"^(team\s+leader|procurement\s+expert|documentation\s+expert"
    r"|urban\s+planning|environmental\s+expert|animal\s+care"
    r"|ict\s*/\s*it|gis\s+expert|data\s+analyst|legal\s+policy"
    r"|urban\s+finance|reporting\s+manager|liaison\s+officer"
    r"|ppp\s+specialist|social\s+development|capacity\s+building"
    r"|financial\s+expert|monitoring\s+expert|procurement\s+specialist)",
    re.IGNORECASE,
)

# ToR action sentences masquerading as parameter names
_TOR_ACTION = re.compile(
    r"^(assist\b|monitor\b|monitoring\b|submission\b|submit\b|prepare\b"
    r"|coordinate\b|ensure\b|must\s+be\s+able|the\s+consultant\s+shall)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_marking_scheme(rfp_doc_name: str) -> dict:
    """
    Returns:
        {
            "criteria":               list[{item_code, parameter, max_marks, criteria_text}],
            "doc_max":                int,
            "grand_total_marks":      int,
            "qualification_threshold": float,
            "eval_pages":             (start, end),
            "schema_warning":         str | None,
            "error":                  str | None,
        }
    """
    pdf_path = UPLOAD_DIR / rfp_doc_name
    if not pdf_path.exists():
        return _err(f"File not found: {pdf_path}")

    print(f"[Step1] Extracting marking scheme from: {rfp_doc_name}")
    doc = fitz.open(str(pdf_path))

    try:
        start_pg, end_pg = _find_eval_pages(doc)
        print(f"[Step1] Evaluation section: p{start_pg} → p{end_pg}")

        lo = max(1, start_pg - 1)
        hi = min(len(doc), end_pg + 1)

        # Primary: geometry-based extraction
        words      = _collect_words(doc, lo, hi)
        page_width = doc[lo - 1].rect.width
        criteria   = _extract_rows(words, page_width, lo, hi)

        # Secondary: text-line fallback if geometry produced nothing
        if not criteria:
            print("[Step1] Geometry produced 0 rows — trying text-line fallback")
            criteria = _text_line_fallback(doc, lo, hi)

        # Validate and clean
        valid = _validate(criteria)

        if not valid:
            return _err("No valid criteria extracted — check eval page range manually")

        doc_max = sum(c["max_marks"] for c in valid)
        print(f"[Step1] Extracted {len(valid)} criteria | doc_max={doc_max}")
        for c in valid:
            print(f"  [{c['item_code']:3s}] {c['parameter'][:55]:55s}  {c['max_marks']:3d} marks")

        schema_warning = None
        marks_list = [c["max_marks"] for c in valid]
        if len(marks_list) < 3:
            schema_warning = f"Only {len(valid)} criteria extracted — likely missed rows, verify manually."

        return {
            "criteria":               valid,
            "doc_max":                doc_max,
            "grand_total_marks":      doc_max,
            "qualification_threshold": 70.0,
            "eval_pages":             (start_pg, end_pg),
            "schema_warning":         schema_warning,
            "error":                  None,
        }

    finally:
        doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Find evaluation pages via TOC + body scan
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_KW = re.compile(
    r"(criteria\s+for\s+(technical\s+)?evaluation"
    r"|evaluation\s+(of\s+)?criteria"
    r"|evaluation\s+of\s+technical\s+bid"
    r"|technical\s+bid\s+eval(uation)?"
    r"|scoring\s+criteria)",
    re.IGNORECASE,
)
_NEXT_KW = re.compile(
    r"(short.?list(ing)?|evaluation\s+of\s+financial|financial\s+bid\s+eval"
    r"|combined\s+and\s+final|general\s+conditions|fraud\s+and\s+corrupt)",
    re.IGNORECASE,
)


def _trailing_page(line: str) -> Optional[int]:
    """Extract the last integer from a TOC line (handles any-length dot leaders)."""
    m = re.search(r'\b(\d{1,3})\s*$', line.rstrip())
    return int(m.group(1)) if m else None


def _find_eval_pages(doc: fitz.Document) -> tuple[int, int]:
    toc_text = ""
    for i in range(min(20, len(doc))):
        toc_text += doc[i].get_text() + "\n"

    lines    = toc_text.splitlines()
    start_pg = end_pg = None

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if start_pg is None and _EVAL_KW.search(line):
            pg = _trailing_page(line)
            if pg and 10 <= pg <= 200:
                start_pg = pg
                for j in range(i + 1, min(i + 15, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        continue
                    if _NEXT_KW.search(nxt):
                        ep = _trailing_page(nxt)
                        if ep and ep >= start_pg:
                            end_pg = ep
                        break
                    ep = _trailing_page(nxt)
                    if ep and ep > start_pg:
                        end_pg = ep
                        break
                break

    if start_pg is None:
        print("[Step1] TOC scan failed — scanning body pages for S.No + marks")
        for pno in range(len(doc)):
            txt = doc[pno].get_text()
            if _EVAL_KW.search(txt) and re.search(r'S\s*[\.\s]*No', txt, re.I):
                start_pg = pno + 1
                end_pg   = min(pno + 6, len(doc))
                print(f"[Step1] Body fallback: eval table at p{start_pg}")
                break

    start_pg = start_pg or 43
    end_pg   = end_pg   or (start_pg + 5)
    return start_pg, end_pg


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Collect word bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def _collect_words(doc: fitz.Document, lo: int, hi: int) -> list[dict]:
    words = []
    for pg_idx in range(lo, hi + 1):
        page = doc[pg_idx - 1]
        for w in page.get_text("words"):      # (x0,y0,x1,y1,word,blk,ln,wrd)
            words.append({
                "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
                "text": w[4], "page": pg_idx,
            })
    return words


# ─────────────────────────────────────────────────────────────────────────────
# Step C: Calibrate column positions from the table header row
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_columns(words: list[dict], page_width: float) -> dict:
    """
    Find X positions of S.No, Parameter, Criteria, and Max-Marks columns
    by locating the table HEADER row words.

    Returns a dict with column boundary X values.
    """
    # ── Max Marks column: find the word "Max" in the right 35% of the page ──
    right_zone = page_width * 0.65
    max_candidates = [
        w for w in words
        if w["x0"] >= right_zone and re.match(r'^[Mm]ax', w["text"])
    ]
    marks_x = min(c["x0"] for c in max_candidates) if max_candidates else page_width * 0.70

    # ── S.No column: find "S." near "No" (the table header) ─────────────────
    # Look for word "S" or "S." in the left half of the page
    sno_x = None
    left_half = page_width * 0.30
    s_words = [w for w in words if re.match(r'^S\.?$', w["text"]) and w["x0"] < left_half]

    for sw in s_words:
        # Check if there is a "No" word within 30px horizontally and 10px vertically
        for ow in words:
            if (re.match(r'^No\.?$', ow["text"], re.I)
                    and abs(ow["y0"] - sw["y0"]) < 12
                    and 0 <= ow["x0"] - sw["x1"] < 30):
                sno_x = sw["x0"]
                break
        if sno_x is not None:
            break

    # If S.No header not found, use leftmost consistent text cluster
    if sno_x is None:
        # Gather all X positions of words in left 20% — the mode is likely the S.No column
        left_xs = [w["x0"] for w in words if w["x0"] < page_width * 0.20]
        if left_xs:
            # Round to nearest 5 and find mode
            from collections import Counter
            rounded = [round(x / 5) * 5 for x in left_xs]
            sno_x = Counter(rounded).most_common(1)[0][0]
        else:
            sno_x = page_width * 0.07  # hard fallback

    # ── Parameter Name column: right of S.No, typically 10-40% of page ──────
    param_x_lo = sno_x + 5
    param_x_hi = page_width * 0.42

    # ── Criteria text column: 40-68% of page ─────────────────────────────────
    crit_x_lo = page_width * 0.40
    crit_x_hi = marks_x - 8

    sno_x_limit = min(sno_x + 35, page_width * 0.20)  # search window for S.No digits

    cols = {
        "sno_x":        sno_x,
        "sno_x_limit":  sno_x_limit,
        "param_x_lo":   param_x_lo,
        "param_x_hi":   param_x_hi,
        "crit_x_lo":    crit_x_lo,
        "crit_x_hi":    crit_x_hi,
        "marks_x":      marks_x,
        "marks_tol":    50,
        "page_width":   page_width,
    }
    print(f"[Step1] Columns: sno_x≈{sno_x:.0f}, marks_x≈{marks_x:.0f}  (page={page_width:.0f})")
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# Step D: Find S.No anchors using calibrated column
# ─────────────────────────────────────────────────────────────────────────────

def _find_sno_anchors(words: list[dict], cols: dict) -> list[dict]:
    """
    A valid S.No anchor is a word that:
      - Is a standalone integer 1-20
      - Lives within the S.No column X range (calibrated ± tolerance)
      - Is NOT at the top of the page in a page-number position
    """
    x_min = max(0, cols["sno_x"] - 10)
    x_max = cols["sno_x_limit"]

    seen  = set()
    found = []

    for w in words:
        if not (x_min <= w["x0"] <= x_max):
            continue
        txt = w["text"].rstrip(".")
        if not re.match(r'^\d{1,2}$', txt):
            continue
        val = int(txt)
        if not (1 <= val <= 20):
            continue
        # Skip if it's near the very top (likely a page number or header)
        if w["y0"] < 50:
            continue
        key = (w["page"], val)
        if key not in seen:
            seen.add(key)
            found.append({**w, "sno": val})

    found.sort(key=lambda a: (a["page"], a["y0"]))

    # Keep first occurrence of each S.No value
    seen_sno: set = set()
    unique = []
    for a in found:
        if a["sno"] not in seen_sno:
            seen_sno.add(a["sno"])
            unique.append(a)

    print(f"[Step1] S.No anchors: {[a['sno'] for a in unique]} "
          f"(searched x={x_min:.0f}–{x_max:.0f})")
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Step E: Extract one row per anchor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rows(words: list[dict], page_width: float, lo: int, hi: int) -> list[dict]:
    cols    = _calibrate_columns(words, page_width)
    anchors = _find_sno_anchors(words, cols)

    if not anchors:
        print("[Step1] 0 S.No anchors — geometry extraction skipped")
        return []

    rows = []
    for i, anchor in enumerate(anchors):
        sno     = anchor["sno"]
        pg      = anchor["page"]
        y_start = anchor["y0"] - 4

        if i + 1 < len(anchors):
            next_a  = anchors[i + 1]
            y_end   = next_a["y0"] - 4
            next_pg = next_a["page"]
        else:
            y_end   = 9999
            next_pg = hi + 1

        # Collect words in this row's region
        row_words = []
        for w in words:
            if w["page"] < pg:
                continue
            if w["page"] > next_pg:
                break
            if w["page"] == pg and w["y0"] < y_start:
                continue
            if w["page"] == next_pg and w["y0"] >= y_end:
                break
            row_words.append(w)

        # ── Max marks: standalone integer in the marks column ────────────────
        marks_x   = cols["marks_x"]
        marks_tol = cols["marks_tol"]

        marks_candidates = [
            w for w in row_words
            if abs(w["x0"] - marks_x) <= marks_tol
            and re.match(r'^\d{1,3}$', w["text"])
            and 1 <= int(w["text"]) <= 100
            # Must NOT be a year or large quantity
            and int(w["text"]) <= 60
        ]

        if not marks_candidates:
            # Widen search to right 32% of page
            marks_candidates = [
                w for w in row_words
                if w["x0"] >= page_width * 0.68
                and re.match(r'^\d{1,3}$', w["text"])
                and 5 <= int(w["text"]) <= 60
            ]

        if not marks_candidates:
            print(f"[Step1] S.No {sno}: no max marks found — skipping")
            continue

        marks_candidates.sort(key=lambda w: (w["page"], w["y0"]))
        max_marks = int(marks_candidates[0]["text"])

        # ── Parameter name: 2nd column, first few lines of the row ───────────
        param_words = sorted(
            [w for w in row_words
             if cols["param_x_lo"] <= w["x0"] <= cols["param_x_hi"]
             and w["page"] == pg
             and w["y0"] <= y_start + 60],
            key=lambda w: (w["y0"], w["x0"]),
        )
        parameter = " ".join(w["text"] for w in param_words).strip()

        # ── Criteria text: middle column ──────────────────────────────────────
        crit_words = sorted(
            [w for w in row_words
             if cols["crit_x_lo"] <= w["x0"] <= cols["crit_x_hi"]],
            key=lambda w: (w["page"], w["y0"], w["x0"]),
        )
        criteria_text = " ".join(w["text"] for w in crit_words).strip()

        rows.append({
            "item_code":     str(sno),
            "parameter":     parameter,
            "max_marks":     max_marks,
            "criteria_text": criteria_text,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Text-line fallback
# ─────────────────────────────────────────────────────────────────────────────

def _text_line_fallback(doc: fitz.Document, lo: int, hi: int) -> list[dict]:
    """
    Reconstruct scoring table from reading-order text.

    Strategy:
      1. Get text per page in reading order.
      2. Split into lines.
      3. Lines starting with a bare digit 1-20 begin a new row.
      4. Accumulate lines until the next row starts.
      5. Extract max_marks = last standalone integer ≤ 60 in the row block.
      6. Extract parameter = first meaningful text after the S.No digit.
    """
    print("[Step1] Text-line fallback: reconstructing from reading-order text")

    full_lines: list[tuple[int, str]] = []   # (page_no, line_text)
    for pg_idx in range(lo, hi + 1):
        if pg_idx > len(doc):
            break
        page_text = doc[pg_idx - 1].get_text("text")
        for line in page_text.splitlines():
            full_lines.append((pg_idx, line))

    rows   = []
    i      = 0
    sno_re = re.compile(r'^\s*(\d{1,2})[\.\s]\s*\S')

    while i < len(full_lines):
        pg, line = full_lines[i]
        m = sno_re.match(line)
        if m:
            sno = int(m.group(1))
            if 1 <= sno <= 20:
                # Collect lines for this row
                block_lines = [line.strip()]
                j = i + 1
                while j < len(full_lines) and j < i + 35:
                    _, nxt = full_lines[j]
                    if sno_re.match(nxt) and int(sno_re.match(nxt).group(1)) != sno:
                        break
                    block_lines.append(nxt.strip())
                    j += 1

                block_text = " ".join(block_lines)

                # Max marks: last integer 5-60 that is followed by "mark" or is standalone
                max_marks = _find_max_marks_in_block(block_text)

                # Parameter: remove S.No prefix, take first 60 chars of useful text
                cleaned = re.sub(r'^\s*\d{1,2}[\.\s]+', '', line).strip()
                parameter = re.split(r'\s{3,}', cleaned)[0][:80].strip()

                # Criteria text: everything except the S.No and max marks marker
                criteria_text = _extract_criteria_text(block_text, max_marks)

                if max_marks and parameter:
                    rows.append({
                        "item_code":     str(sno),
                        "parameter":     parameter,
                        "max_marks":     max_marks,
                        "criteria_text": criteria_text,
                    })

                i = j
                continue
        i += 1

    return rows


def _find_max_marks_in_block(text: str) -> Optional[int]:
    """
    Find the max marks integer from a row block.
    Priority: standalone integer adjacent to 'mark' keyword.
    Falls back to the largest standalone integer 5-60.
    """
    # Best: integer immediately before/after 'marks'
    m = re.search(r'(\d+)\s*mark\s*s?\b', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 5 <= val <= 60:
            return val

    # Fallback: largest standalone integer in range 5-60
    # (avoid years, percentages near %, crores)
    candidates = []
    for match in re.finditer(r'\b(\d{1,3})\b', text):
        val = int(match.group(1))
        if 5 <= val <= 60:
            # Make sure it's not a year (2020-2025)
            if 2000 <= val <= 2030:
                continue
            # Make sure it's not followed by Cr/crore/lakh/%
            after = text[match.end():match.end() + 6].lower()
            if re.match(r'\s*(?:cr|lakh|%|rs)', after):
                continue
            candidates.append(val)

    return max(candidates) if candidates else None


def _extract_criteria_text(block_text: str, max_marks: Optional[int]) -> str:
    """Remove the max marks marker and clean up the block for criteria text."""
    text = block_text
    if max_marks:
        # Remove "N marks" patterns
        text = re.sub(
            rf'\b{re.escape(str(max_marks))}\s*mark\s*s?\b', '', text, flags=re.IGNORECASE
        )
    # Remove S.No prefix
    text = re.sub(r'^\s*\d{1,2}[\.\s]+', '', text)
    return " ".join(text.split()).strip()[:1000]


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate(criteria: list[dict]) -> list[dict]:
    """Drop invalid, presentation-only, ToR-action, and sub-item rows."""
    valid = []
    for c in criteria:
        p  = c.get("parameter", "").strip()
        mm = int(c.get("max_marks") or 0)

        if not p or mm < 1:
            continue
        if mm > 100:
            print(f"[Step1] Dropped (marks={mm} > 100): {p[:60]}")
            continue
        if _SKIP_PARAMS.search(p):
            print(f"[Step1] Dropped (skip pattern): {p[:60]}")
            continue
        if _SUBITEM_ROLE.match(p):
            print(f"[Step1] Dropped (sub-item role): {p[:60]}")
            continue
        if _TOR_ACTION.match(p):
            print(f"[Step1] Dropped (ToR action): {p[:60]}")
            continue
        # Very small marks (1-4) with no scoring keyword = sub-item allocation
        if mm <= 4 and not re.search(
            r"(turnover|experience|qualification|methodology|competence|personnel|manpower)",
            p, re.IGNORECASE,
        ):
            print(f"[Step1] Dropped (tiny marks={mm}, no criteria keyword): {p[:60]}")
            continue

        valid.append(c)

    # Deduplicate: same (parameter_lower, max_marks) keeps longest criteria_text
    seen: dict = {}
    for c in valid:
        key = (re.sub(r"\s+", " ", c["parameter"]).strip().lower(), c["max_marks"])
        if key not in seen or len(c["criteria_text"]) > len(seen[key]["criteria_text"]):
            seen[key] = c

    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _err(msg: str) -> dict:
    print(f"[Step1] ERROR: {msg}")
    return {
        "criteria": [], "doc_max": 0, "grand_total_marks": 0,
        "qualification_threshold": 70.0, "eval_pages": (0, 0),
        "schema_warning": None, "error": msg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    doc_name = sys.argv[1] if len(sys.argv) > 1 else "7b21dfca.pdf"
    result   = extract_marking_scheme(doc_name)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))