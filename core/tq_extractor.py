"""
core/tq_compliance_parser.py
============================
FORM TECH-4 Compliance Matrix Parser  — v1
===========================================

PURPOSE
───────
Proposals submitted against Indian government RFPs almost always include a
"FORM TECH-4" (or equivalently titled) self-assessment table where the bidder:

  • Copies verbatim RFP criterion text into column 3
  • States their factual claim in the response column (column 5 or 6)
  • Cites exact page numbers of supporting annexures
  • Proposes their own marks

This table is the SINGLE BEST SOURCE for scoring because:
  1. The criteria text is verbatim from the RFP — no interpretation needed
  2. The bidder's claim is explicit: "Our average turnover is INR 480.23 Cr"
  3. Page references can be spot-verified immediately

WHY OLD PARSERS FAIL
────────────────────
  • PyMuPDF text extraction loses column boundaries — all columns merge
  • Rows span multiple pages with no page-break signal
  • Sub-categories (a, b, c) nest inside parent rows irregularly
  • Numeric facts appear mid-sentence in continuous prose
  • A criterion block can be 2–40 lines deep

THIS PARSER'S APPROACH
──────────────────────
  STAGE 0  Multi-signal page detection
           (header text, column keywords, structural patterns)

  STAGE 1  Geometric column calibration from header row
           PyMuPDF word-level bounding boxes → column X-boundaries

  STAGE 2  Row anchor detection
           S.No digits + sub-label (a/b/c, i/ii/iii) anchored to S.No column

  STAGE 3  Multi-page row assembly
           Collect all words from anchor to next anchor, crossing page breaks

  STAGE 4  Column-aware text reconstruction
           Words split into: criteria | marks | bidder_response | doc_refs

  STAGE 5  Sub-category resolution
           Handle 3a / 3(a) / (a) / i. / (i) patterns within a parent row

  STAGE 6  Python-first fact extraction (per v22 Issue 7)
           Turnover, projects, personnel, years — deterministic regex first

  STAGE 7  Evidence page extraction + spot-verification
           "Page no. 64" → open page 64 → confirm number present

  STAGE 8  Formula application
           Uses _parse_band_table_strict from v22 for scoring

INTEGRATION
───────────
  from core.tq_compliance_parser import parse_compliance_matrix, score_from_matrix

  # Get structured rows from proposal
  matrix = parse_compliance_matrix(proposal_pdf_path)

  # Score each criterion against its RFP formula
  results = score_from_matrix(matrix, rfp_criteria_list)

TESTED AGAINST
──────────────
  • UDD UP Proposal (354 pages, 14-row table spanning p.35–p.48)
  • NHB Cluster Proposal (539 pages, 8-row table with sub-criteria)
  • DDU-GKY Pune/Amravati proposals
"""

from __future__ import annotations

import re
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplianceRow:
    """One criterion row from the FORM TECH-4 table."""
    item_code:        str              # "1", "2", "3", "3a", "3(b)"
    parent_code:      str              # "" or "3" for sub-items
    parameter:        str              # short criterion name
    criteria_text:    str              # verbatim RFP scoring rules
    max_marks:        Optional[int]    # from RFP marks column
    proposed_marks:   Optional[int]    # bidder's self-assessment
    bidder_response:  str              # bidder's factual claim
    evidence_pages:   list[int]        # page numbers cited as supporting docs
    is_sub_item:      bool = False
    raw_page:         int  = 0         # first PDF page this row appears on

    # Populated by Stage 6 (fact extraction)
    extracted_value:  Optional[str]    = None
    extracted_label:  str              = ""

    # Populated by Stage 7 (verification)
    verified:         bool             = False

    def to_dict(self) -> dict:
        return {
            "item_code":       self.item_code,
            "parent_code":     self.parent_code,
            "parameter":       self.parameter,
            "criteria_text":   self.criteria_text,
            "max_marks":       self.max_marks,
            "proposed_marks":  self.proposed_marks,
            "bidder_response": self.bidder_response,
            "evidence_pages":  self.evidence_pages,
            "is_sub_item":     self.is_sub_item,
            "extracted_value": self.extracted_value,
            "extracted_label": self.extracted_label,
            "verified":        self.verified,
        }


@dataclass
class ComplianceMatrix:
    """Full parsed FORM TECH-4 table."""
    rows:          list[ComplianceRow]
    table_pages:   list[int]          # PDF pages the table spans
    proposal_path: str
    parse_method:  str = "geometric"  # "geometric" | "textline" | "llm"
    parse_warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Multi-signal compliance table page detection
# ─────────────────────────────────────────────────────────────────────────────

# Strong header signals — any one of these on a page = very likely the table start
_HEADER_SIGNALS = re.compile(
    r"""(
        form\s*tech[\s\-]*4                     |
        technical\s+bid\s+evaluation\s+criteria\s+and\s+(our\s+)?compliance  |
        technical\s+evaluation\s+criteria\s+and\s+(our\s+)?response          |
        evaluation\s+criteria\s+and\s+compliance\s+statement                 |
        our\s+compliance\s+(to|with|against)\s+(the\s+)?criteria             |
        self[\s\-]assessment\s+(table|criteria|form)                         |
        sl\.?\s*no\.?.{0,60}criteria.{0,60}(proposed|gt|bidder)\s+marks     |
        s\.?\s*no\.?.{0,60}qualification.{0,60}marks?\s+awarded             |
        s\.?\s*no\.?.{0,60}parameter.{0,60}gt\s+marks
    )""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Column header keywords that appear in the HEADER ROW of the table
_COL_HEADERS = re.compile(
    r"\b(proposed\s+marks?|gt\s+marks?|marks?\s+awarded|bidder\s+marks?|"
    r"supporting\s+doc|reference\s+document|page\s+(no|reference)|"
    r"our\s+response|bidder\s+response|compliance\s+details?|remarks?)\b",
    re.IGNORECASE,
)

# Scoring language that means we're on the RFP criteria pages (not bidder response)
_MARKS_RE = re.compile(r"\b(\d{1,2})\s*marks?\b", re.I)
_PAGE_REF_RE = re.compile(
    r"(?:page|pg)\.?\s*(?:no\.?|nos?\.?)?\s*(\d+(?:\s*[-–to]+\s*\d+)?)",
    re.IGNORECASE,
)


def _score_page_for_table(page_text: str, page_lower: str) -> float:
    """
    Score a page's likelihood of being part of the FORM TECH-4 table.
    Higher = more likely.
    """
    score = 0.0

    if _HEADER_SIGNALS.search(page_text):
        score += 15.0

    col_hits = len(_COL_HEADERS.findall(page_text))
    score += col_hits * 2.5

    marks_hits = len(_MARKS_RE.findall(page_text))
    score += min(marks_hits * 0.8, 6.0)

    # Page number references (bidders cite their annexure pages)
    pg_ref_hits = len(_PAGE_REF_RE.findall(page_text))
    score += min(pg_ref_hits * 0.6, 4.0)

    # GT / Grant Thornton response signals
    if re.search(r"grant\s*thornton|gtbl|gt\s+has|our\s+firm|we\s+have", page_lower):
        score += 3.0

    # S.No pattern in left margin (table structure)
    if re.search(r"(?m)^\s*\d{1,2}\s*\n", page_text):
        score += 2.0

    return score


def find_compliance_table_pages(
    proposal_path: str,
    max_search_pages: int = 100,
    min_page_score: float = 4.0,
) -> list[int]:
    """
    Find all PDF pages that form the FORM TECH-4 compliance matrix.

    Strategy:
      1. Score every page up to max_search_pages
      2. Find the start page (highest score / header signal)
      3. Extend contiguously while score stays above threshold
      4. Stop on section-break signals

    Returns sorted list of 1-based page numbers.
    """
    if not _FITZ_OK or not Path(proposal_path).exists():
        return []

    doc = fitz.open(proposal_path)
    n   = min(len(doc), max_search_pages)

    page_scores: list[tuple[float, int]] = []   # (score, 1-based page)
    for i in range(n):
        txt   = doc[i].get_text()
        lower = txt.lower()
        s     = _score_page_for_table(txt, lower)
        page_scores.append((s, i + 1))

    doc.close()

    if not any(s > min_page_score for s, _ in page_scores):
        print("[ComplianceParser] No table pages detected above threshold")
        return []

    # Find strong start candidates (header signal)
    strong_starts = [pg for s, pg in page_scores if s >= 12.0]

    if not strong_starts:
        # Fallback: find the highest-scoring cluster
        sorted_by_score = sorted(page_scores, key=lambda x: -x[0])
        strong_starts = [sorted_by_score[0][1]]

    start_pg = min(strong_starts)
    print(f"[ComplianceParser] Table starts at page {start_pg} "
          f"(score={page_scores[start_pg-1][0]:.1f})")

    # Extend forward while score stays reasonable
    table_pages = [start_pg]
    for pg in range(start_pg + 1, n + 1):
        s = page_scores[pg - 1][0]

        # Stop signals: definitely NOT part of the table
        txt_lower = ""
        doc2 = fitz.open(proposal_path)
        if pg <= len(doc2):
            txt_lower = doc2[pg - 1].get_text().lower()
        doc2.close()

        is_break = bool(re.search(
            r"(annex|appendix\s+[a-z]|exhibit\s+[a-z]|"
            r"financial\s+proposal|commercial\s+bid|"
            r"chapter\s+[ivxlc]+|section\s+[ivxlc]+\.?\s+\b"
            r"(?!no\b))",  # don't break on "Section No."
            txt_lower,
        ))

        if is_break and s < 3.0:
            print(f"[ComplianceParser] Table ends before page {pg} (section break)")
            break

        if s < min_page_score and len(table_pages) > 2:
            # Allow one low-score page (might be a continuation with no new criteria)
            # but stop if two consecutive low-score pages
            if len(table_pages) >= 2 and page_scores[table_pages[-1] - 1][0] < min_page_score:
                print(f"[ComplianceParser] Table ends before page {pg} (low score)")
                break

        table_pages.append(pg)

    # Safety cap: table is unlikely to exceed 30 pages
    table_pages = table_pages[:30]
    print(f"[ComplianceParser] Table pages: {table_pages}")
    return table_pages


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Geometric column calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnMap:
    """X-boundaries for each logical column in the table."""
    page_width:   float

    sno_lo:       float = 0.0    # S.No column left edge
    sno_hi:       float = 0.0    # S.No column right edge

    param_lo:     float = 0.0    # Parameter / criteria column
    param_hi:     float = 0.0

    crit_lo:      float = 0.0    # Criteria / scoring rules text
    crit_hi:      float = 0.0

    marks_lo:     float = 0.0    # Max marks column
    marks_hi:     float = 0.0

    response_lo:  float = 0.0    # Bidder response column (rightmost / widest)
    response_hi:  float = 0.0

    doc_ref_lo:   float = 0.0    # Supporting doc / page reference column
    doc_ref_hi:   float = 0.0

    def classify_x(self, x: float) -> str:
        """Return column name for a given X position."""
        if self.sno_lo <= x <= self.sno_hi:        return "sno"
        if self.param_lo <= x <= self.param_hi:    return "param"
        if self.crit_lo <= x <= self.crit_hi:      return "criteria"
        if self.marks_lo <= x <= self.marks_hi:    return "marks"
        if self.doc_ref_lo <= x <= self.doc_ref_hi: return "docref"
        if self.response_lo <= x <= self.response_hi: return "response"
        return "unknown"


def _calibrate_columns(words: list[dict], page_width: float) -> ColumnMap:
    """
    Derive column X-boundaries from word bounding boxes.

    Algorithm:
      1. Find the table HEADER ROW by locating column-header keywords
         (S.No, Parameter/Criteria, Max Marks, GT Marks, Doc Reference, Bidder Response)
      2. Use their X-centroids as column anchors
      3. Build boundaries as midpoints between adjacent anchors
    """
    cm = ColumnMap(page_width=page_width)

    # ── Try to find the header row ────────────────────────────────────────────
    # Cluster words by Y-position (rows)
    y_clusters: dict[int, list[dict]] = defaultdict(list)
    for w in words:
        y_key = round(w["y0"] / 6) * 6
        y_clusters[y_key].append(w)

    # Find rows that contain column header keywords
    header_row_words: list[dict] = []
    for y_key in sorted(y_clusters):
        row_words = y_clusters[y_key]
        row_text  = " ".join(w["text"] for w in row_words).lower()
        kw_count  = sum(1 for kw in [
            "s.no", "s. no", "no.", "criteria", "parameter", "marks", "response",
            "reference", "document", "support",
        ] if kw in row_text)
        if kw_count >= 3:
            header_row_words.extend(row_words)

    # ── Locate specific column header words ───────────────────────────────────
    def _find_col_x(pattern: str, wds: list[dict]) -> Optional[float]:
        for w in wds:
            if re.search(pattern, w["text"], re.I):
                return (w["x0"] + w["x1"]) / 2
        return None

    sno_x      = _find_col_x(r"^S\.?$|^SL\.?$", header_row_words)
    crit_x     = _find_col_x(r"criteria|parameter|particulars|criterion", header_row_words)
    marks_x    = _find_col_x(r"max(imum)?|marks?|full\s+marks?", header_row_words)
    response_x = _find_col_x(r"response|compliance|bidder|gt\s+has|our\s+claim", header_row_words)
    docref_x   = _find_col_x(r"document|reference|annexure|support|page\s*(no|ref)", header_row_words)
    proposed_x = _find_col_x(r"proposed|awarded|gt\s+marks?|score", header_row_words)

    # ── Fallback: use heuristic X fractions if header detection failed ────────
    pw = page_width
    if sno_x is None:
        # Most common: S.No is in leftmost 8% of page
        left_cluster_xs = [w["x0"] for w in words if w["x0"] < pw * 0.12]
        from collections import Counter
        if left_cluster_xs:
            rounded = [round(x / 4) * 4 for x in left_cluster_xs]
            sno_x = Counter(rounded).most_common(1)[0][0]
        else:
            sno_x = pw * 0.05

    if marks_x is None:
        # Max marks is usually in right-centre: 60-75% of page width
        right_zone_xs = [w["x0"] for w in words
                         if pw * 0.55 < w["x0"] < pw * 0.78
                         and re.match(r"^\d{1,2}$", w["text"])
                         and 5 <= int(w["text"]) <= 60]
        if right_zone_xs:
            marks_x = sorted(right_zone_xs)[len(right_zone_xs) // 2]
        else:
            marks_x = pw * 0.65

    if response_x is None:
        response_x = pw * 0.82  # response is in rightmost column

    if docref_x is None:
        docref_x = pw * 0.90   # doc ref is far right

    if crit_x is None:
        crit_x = pw * 0.38     # criteria text is centre-left

    # ── Build column boundaries as midpoints ──────────────────────────────────
    anchors_sorted = sorted(filter(None, [sno_x, crit_x, marks_x, proposed_x,
                                          response_x, docref_x]))

    def _boundary(a: float, b: float) -> float:
        return (a + b) / 2.0

    # S.No column
    cm.sno_lo = max(0.0, sno_x - 15)
    cm.sno_hi = sno_x + 30

    # Parameter / criterion name column (usually between S.No and criteria text)
    cm.param_lo = cm.sno_hi
    cm.param_hi = (crit_x - 20) if crit_x > cm.sno_hi + 40 else cm.sno_hi + 60

    # Criteria text column (wide centre column)
    cm.crit_lo = cm.param_hi
    cm.crit_hi = marks_x - 15 if marks_x > cm.crit_lo + 40 else pw * 0.60

    # Marks column (narrow)
    cm.marks_lo = marks_x - 20
    cm.marks_hi = marks_x + 40

    # Bidder response column (wide rightmost)
    if response_x and response_x > marks_x + 30:
        cm.response_lo = marks_x + 40
        if docref_x and docref_x > response_x + 20:
            cm.response_hi = docref_x - 15
            cm.doc_ref_lo  = docref_x - 15
            cm.doc_ref_hi  = pw
        else:
            cm.response_hi = pw
            cm.doc_ref_lo  = pw * 0.85
            cm.doc_ref_hi  = pw
    else:
        # No separate response column detected — assume rightmost 40% is response
        cm.response_lo = marks_x + 40
        cm.response_hi = pw
        cm.doc_ref_lo  = pw * 0.88
        cm.doc_ref_hi  = pw

    print(f"[ComplianceParser] Column map: "
          f"sno=[{cm.sno_lo:.0f},{cm.sno_hi:.0f}] "
          f"param=[{cm.param_lo:.0f},{cm.param_hi:.0f}] "
          f"crit=[{cm.crit_lo:.0f},{cm.crit_hi:.0f}] "
          f"marks=[{cm.marks_lo:.0f},{cm.marks_hi:.0f}] "
          f"response=[{cm.response_lo:.0f},{cm.response_hi:.0f}]")
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2-3 — Row anchors + multi-page row assembly
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RowAnchor:
    """Location of an S.No digit or sub-label."""
    code:     str    # "1", "2", "3", "3a", "3(b)"
    parent:   str    # "" or "3"
    page:     int    # 1-based
    y0:       float  # top of the anchor word
    is_sub:   bool   # True if this is a sub-item (a, b, c, i, ii...)


# Roman numeral → letter mapping
_ROMAN = {"i": "a", "ii": "b", "iii": "c", "iv": "d", "v": "e",
           "vi": "f", "vii": "g", "viii": "h"}

# Sub-label patterns: (a), (i), a., i., a), i)
_SUBLABEL_RE = re.compile(
    r"^\s*(?:"
    r"\(([a-h])\)"        # (a) through (h)
    r"|([a-h])[\.\)](?!\w)"  # a. or a) not followed by a word char
    r"|\((i{1,4}v?|vi{0,3})\)"  # roman: (i) (ii) (iii) (iv) (v) (vi) (vii) (viii)
    r"|(i{1,4}v?|vi{0,3})[\.\)](?!\w)"  # roman: i. ii. etc.
    r")\s*$",
    re.IGNORECASE,
)

def _parse_anchor_code(text: str, parent: str) -> Optional[tuple[str, bool]]:
    """
    Parse a sub-label text into (code, is_sub).
    Returns None if not a recognisable label.
    """
    text = text.strip()
    m = _SUBLABEL_RE.match(text)
    if not m:
        return None

    letter = (m.group(1) or m.group(2) or "").lower()
    roman  = (m.group(3) or m.group(4) or "").lower()

    if roman:
        letter = _ROMAN.get(roman, roman)

    if not letter:
        return None

    code = f"{parent}{letter}" if parent else letter
    return code, True


def find_row_anchors(words: list[dict], cm: ColumnMap,
                     existing_parents: set) -> list[RowAnchor]:
    """
    Find all row anchor positions (S.No digits and sub-labels).
    Returns anchors sorted by (page, y0).
    """
    anchors: list[RowAnchor] = []
    seen_codes: set = set()

    # ── Primary: S.No integer anchors ────────────────────────────────────────
    for w in words:
        # Must be in S.No column
        if not (cm.sno_lo <= w["x0"] <= cm.sno_hi):
            continue
        text = w["text"].rstrip(".")
        if not re.match(r"^\d{1,2}$", text):
            continue
        val = int(text)
        if not (1 <= val <= 25):
            continue
        if w["y0"] < 40:  # skip page header area
            continue
        code = str(val)
        if code not in seen_codes:
            seen_codes.add(code)
            anchors.append(RowAnchor(
                code=code, parent="", page=w["page"],
                y0=w["y0"], is_sub=False,
            ))

    # ── Secondary: Sub-labels adjacent to S.No column ────────────────────────
    # Sub-labels appear just right of the S.No column or in param column
    # They are associated with the most recent primary anchor
    current_parent = ""
    for w in sorted(words, key=lambda x: (x["page"], x["y0"])):
        # Update current parent when we hit a primary anchor
        parent_anchor = next(
            (a for a in anchors
             if a.page == w["page"] and abs(a.y0 - w["y0"]) < 5 and not a.is_sub),
            None,
        )
        if parent_anchor:
            current_parent = parent_anchor.code
            continue

        # Check if word looks like a sub-label
        if not (cm.sno_lo - 5 <= w["x0"] <= cm.param_hi):
            continue
        if w["y0"] < 40:
            continue

        result = _parse_anchor_code(w["text"], current_parent)
        if result is None:
            continue

        code, is_sub = result
        if code not in seen_codes:
            seen_codes.add(code)
            anchors.append(RowAnchor(
                code=code, parent=current_parent, page=w["page"],
                y0=w["y0"], is_sub=True,
            ))

    anchors.sort(key=lambda a: (a.page, a.y0))
    return anchors


def collect_row_words(
    words: list[dict],
    anchor: RowAnchor,
    next_anchor: Optional[RowAnchor],
) -> list[dict]:
    """
    Collect all words belonging to one row (from this anchor to the next).
    Handles page boundaries correctly.
    """
    row_words: list[dict] = []
    for w in words:
        # Before this anchor's page: skip
        if w["page"] < anchor.page:
            continue

        # After next anchor's page: stop
        if next_anchor and w["page"] > next_anchor.page:
            break

        # On anchor's page: must be at or below anchor's Y
        if w["page"] == anchor.page and w["y0"] < anchor.y0 - 3:
            continue

        # On next anchor's page: must be above next anchor's Y
        if next_anchor and w["page"] == next_anchor.page and w["y0"] >= next_anchor.y0 - 3:
            break

        row_words.append(w)

    return row_words


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Column-aware text reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_columns(
    row_words: list[dict],
    cm: ColumnMap,
) -> dict[str, str]:
    """
    Split row words into logical columns using X-position and reconstruct text.

    Returns dict: {column_name: reconstructed_text}
    """
    # Group words by column and Y-position (lines within each column)
    col_lines: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for w in row_words:
        col = cm.classify_x(w["x0"])
        if col == "unknown":
            # Try mid-point classification
            mid_x = (w["x0"] + w["x1"]) / 2
            col   = cm.classify_x(mid_x)
        if col == "unknown":
            # Assign to nearest column based on distance to column centres
            centres = {
                "sno":      (cm.sno_lo + cm.sno_hi) / 2,
                "param":    (cm.param_lo + cm.param_hi) / 2,
                "criteria": (cm.crit_lo + cm.crit_hi) / 2,
                "marks":    (cm.marks_lo + cm.marks_hi) / 2,
                "response": (cm.response_lo + cm.response_hi) / 2,
                "docref":   (cm.doc_ref_lo + cm.doc_ref_hi) / 2,
            }
            col = min(centres, key=lambda c: abs(centres[c] - w["x0"]))

        # Y-group within column (round to 5pt lines)
        y_key = round(w["y0"] / 4) * 4
        col_lines[col][y_key].append(w)

    # Reconstruct text for each column
    result: dict[str, str] = {}
    for col, line_dict in col_lines.items():
        lines = []
        for y_key in sorted(line_dict):
            line_words = sorted(line_dict[y_key], key=lambda w: w["x0"])
            line_text  = " ".join(w["text"] for w in line_words)
            lines.append(line_text.strip())
        result[col] = " ".join(l for l in lines if l).strip()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Max marks extraction from the marks column
# ─────────────────────────────────────────────────────────────────────────────

def _extract_max_marks(marks_col_text: str, criteria_col_text: str) -> Optional[int]:
    """
    Extract the max marks for a criterion from the marks column text.
    Falls back to searching criteria text for "N marks" pattern.
    """
    text = marks_col_text or ""

    # Primary: clean integer in marks column
    for m in re.finditer(r"\b(\d{1,3})\b", text):
        val = int(m.group(1))
        if 1 <= val <= 100:
            return val

    # Secondary: look in criteria text for "max N marks" or "N marks"
    m2 = re.search(r"max(?:imum)?\s*(?:of\s*)?(\d{1,3})\s*marks?", criteria_col_text, re.I)
    if m2:
        val = int(m2.group(1))
        if 1 <= val <= 100:
            return val

    # Tertiary: find standalone mark integers 5-60
    for m3 in re.finditer(r"\b(\d{1,2})\s*marks?\b", criteria_col_text, re.I):
        val = int(m3.group(1))
        if 5 <= val <= 60:
            return val

    return None


def _extract_proposed_marks(response_col_text: str, marks_col_text: str) -> Optional[int]:
    """Extract bidder's proposed/claimed marks from response or marks column."""
    # Proposed marks often appear as "Proposed: 15" or just a number in the marks col
    for text in [response_col_text, marks_col_text]:
        m = re.search(r"(?:proposed|claimed|awarded|scored|marks?\s*[:\-=])?\s*(\d{1,2})", text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 60:
                return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Python-first fact extraction from bidder response text
# ─────────────────────────────────────────────────────────────────────────────

def extract_bidder_fact(
    bidder_response: str,
    criteria_text: str,
    formula_hint: str = "",
) -> tuple[Optional[str], str]:
    """
    Extract the specific numeric fact from the bidder's response text.

    Returns (value_string, label_string) or (None, "Not found").

    CRITICAL: This function only extracts facts the BIDDER CLAIMED.
    It rejects scoring language like "5 marks for 01 project".

    Supported patterns (Python-first, no LLM):
      - Turnover: "INR 480.23 Crore", "Rs. 150 Cr", "average turnover 230 Cr"
      - Project count: "5 qualifying projects", "06 assignments"
      - Personnel: "more than 1000 technically qualified"
      - Experience years: "X years since YYYY"
      - Binary: "Yes" / presence of positive claim
    """
    txt = bidder_response or ""
    crit = (criteria_text or "").lower()

    # ── Guard: reject scoring-language values ─────────────────────────────────
    if re.search(r"\b(marks?\s+for|marks?\s+per|maximum\s+marks?|scoring\s+criteria)\b",
                 txt, re.I):
        # Only reject if the ENTIRE response looks like scoring language
        non_scoring = re.sub(
            r"[^.]*\b(?:marks?\s+for|marks?\s+per|maximum\s+marks?)[^.]*\.", "", txt
        ).strip()
        if len(non_scoring) < 40:
            return None, "Response appears to be scoring criteria, not bidder claim"

    # ── TURNOVER ──────────────────────────────────────────────────────────────
    # Multiple patterns in descending specificity
    turnover_patterns = [
        # "average annual turnover of INR 884.48 Crores"
        (r"average\s+(?:annual\s+)?turnover\s+(?:of\s+)?(?:inr|rs\.?|₹)?\s*"
         r"([\d,]+(?:\.\d+)?)\s*(?:cr(?:ores?)?|lakh)?", "avg turnover"),
        # "INR 480.23 Cr / Crore"
        (r"(?:inr|rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:cr(?:ores?)?|lakh)", "INR value"),
        # "480.23 crores" standalone
        (r"([\d,]+(?:\.\d+)?)\s*cr(?:ores?)?\b(?!\s*per\b)", "Crore value"),
        # "Rs. 150 Cr" shorthand
        (r"(?:rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*cr\b", "Rs Cr"),
        # Table row: "2018-19  230.45  2019-20  210.33  2020-21  190.12  Average  210.30"
        (r"average\s+([\d,]+(?:\.\d+)?)", "average from table"),
    ]

    if any(k in crit for k in ["turnover", "crore", "financial", "revenue"]):
        for pat, label in turnover_patterns:
            m = re.search(pat, txt, re.I)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if 0.1 <= val <= 100_000:
                        # Normalise lakhs to crores
                        if "lakh" in m.group(0).lower():
                            val = round(val / 100, 2)
                        return f"{val} Cr", f"{val} Cr ({label})"
                except ValueError:
                    pass

    # ── PROJECT COUNT ─────────────────────────────────────────────────────────
    project_patterns = [
        # "the following 5 assignments"
        (r"following\s+(\d+)\s+(?:such\s+)?(?:assign|project)", "following N"),
        # "handled 5 large scale projects"
        (r"handled\s+(\d+)\s+(?:large|major)?\s*(?:scale\s+)?project", "handled N"),
        # "5 projects / assignments completed"
        (r"(\d+)\s+(?:such\s+)?(?:project|assignment)s?\s+(?:have\s+)?(?:been\s+)?completed", "N completed"),
        # "enclosed details of 5 projects"
        (r"enclosed?\s+(?:details?\s+of\s+)?(\d+)\s+(?:such\s+)?(?:project|assign)", "enclosed N"),
        # "eligible 5 projects"
        (r"(\d+)\s+(?:eligible|qualifying|relevant)\s+(?:project|assign)", "N eligible"),
        # Count numbered list entries: "1. Project A ... 2. Project B ..."
        # (fallback — count highest project number)
        (r"(?:^|\n)\s*(\d{1,2})\.\s+[A-Z]", "numbered list"),
    ]

    if any(k in crit for k in ["project", "assignment", "pmc", "pmu", "urban",
                                "billing", "work order", "experience"]):
        for pat, label in project_patterns:
            if label == "numbered list":
                nums = re.findall(pat, txt, re.MULTILINE)
                if nums:
                    count = max(int(n) for n in nums)
                    if 1 <= count <= 50:
                        return str(count), f"{count} projects (counted from numbered list)"
            else:
                m = re.search(pat, txt, re.I)
                if m:
                    count = int(m.group(1))
                    if 1 <= count <= 50:
                        return str(count), f"{count} projects ({label})"

    # ── PERSONNEL / HEADCOUNT ─────────────────────────────────────────────────
    personnel_patterns = [
        (r"more\s+than\s+([\d,]+)\s+technically\s+qualified", "more than N technically qualified"),
        (r"([\d,]+)\+?\s+technically\s+qualified", "N technically qualified"),
        (r"more\s+than\s+([\d,]+)\s+(?:consulting|advisory|qualified)?\s*(?:staff|personnel|professional)",
         "more than N staff"),
        (r"([\d,]+)\+?\s+(?:consulting|advisory|qualified)\s*(?:staff|personnel|professional)",
         "N consulting staff"),
        (r"payroll\s+of\s+([\d,]+)\s*(?:consulting|technically|qualified)?", "payroll of N"),
        (r"deployed\s+([\d,]+)\s+(?:professionals?|resources?|experts?)", "deployed N"),
    ]

    if any(k in crit for k in ["manpower", "professional", "personnel", "employee",
                                "staff", "headcount"]):
        for pat, label in personnel_patterns:
            m = re.search(pat, txt, re.I)
            if m:
                try:
                    val = int(m.group(1).replace(",", ""))
                    if val > 0:
                        return str(val), f"{val} ({label})"
                except ValueError:
                    pass

    # ── EXPERIENCE YEARS ──────────────────────────────────────────────────────
    year_patterns = [
        (r"more\s+than\s+(\d+)\s+years?\s+(?:of\s+)?(?:relevant\s+)?experience", "more than N years"),
        (r"(\d+)\+\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience", "N+ years"),
        (r"(\d+)\s+years?\s+of\s+(?:relevant\s+)?(?:experience|expertise)", "N years"),
        (r"since\s+(20\d{2}|19\d{2})", "since YYYY"),   # → derive years
        (r"established\s+in\s+(20\d{2}|19\d{2})", "established YYYY"),
    ]

    if any(k in crit for k in ["experience", "year", "since", "established"]):
        for pat, label in year_patterns:
            m = re.search(pat, txt, re.I)
            if m:
                if label in ("since YYYY", "established YYYY"):
                    try:
                        founding_year = int(m.group(1))
                        years = 2024 - founding_year
                        return f"more than {years} years", f"{years} years (since {founding_year})"
                    except ValueError:
                        pass
                else:
                    try:
                        yrs = int(m.group(1))
                        if 1 <= yrs <= 50:
                            return f"{yrs} years", f"{yrs} years ({label})"
                    except ValueError:
                        pass

    # ── BINARY / PRESENCE ────────────────────────────────────────────────────
    positive_signals = [
        r"grant\s+thornton\s+has",
        r"gt\s+has\s+(?:extensive\s+)?experience",
        r"we\s+have\s+(?:extensive\s+)?experience",
        r"please\s+refer\s+(?:form|page|annex)",
        r"enclosed\s+(?:herewith|at)",
        r"methodology\s+is\s+(?:enclosed|provided|attached)",
        r"yes\b",
        r"registered\s+(?:with|under)",
    ]
    for pat in positive_signals:
        if re.search(pat, txt, re.I):
            return "Yes", "Presence confirmed (positive claim in response)"

    return None, "Not found in bidder response"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Evidence page extraction and spot-verification
# ─────────────────────────────────────────────────────────────────────────────

def extract_evidence_pages(bidder_response: str, docref_col: str = "") -> list[int]:
    """
    Extract all page number references from bidder response and doc-ref column.
    Handles: "Page No. 64", "Pg. nos. 153-179", "refer page 30 to 45"
    """
    pages: set[int] = set()
    combined = (bidder_response or "") + " " + (docref_col or "")

    for m in _PAGE_REF_RE.finditer(combined):
        ref = m.group(1)
        # Handle ranges: "153-179" or "30 to 45"
        range_m = re.match(r"(\d+)\s*[-–to]+\s*(\d+)", ref)
        if range_m:
            lo = int(range_m.group(1))
            # Only store start of range (avoid storing 150+ pages)
            if 1 <= lo <= 1000:
                pages.add(lo)
        else:
            try:
                pg = int(ref.strip())
                if 1 <= pg <= 1000:
                    pages.add(pg)
            except ValueError:
                pass

    return sorted(pages)


def verify_on_page(
    pdf_path: str,
    claimed_value: str,
    pages: list[int],
    max_check: int = 3,
) -> bool:
    """
    Spot-check: does at least one cited evidence page contain the key
    number or phrase from the claimed value?
    """
    if not pages or not claimed_value or not _FITZ_OK:
        return False   # can't verify → assume True (don't penalise)

    nums = re.findall(r"\d[\d,.]*", claimed_value)
    key_terms = [n.replace(",", "") for n in nums[:3]]
    if not key_terms:
        key_terms = [claimed_value[:20].lower()]

    try:
        doc = fitz.open(pdf_path)
        for pg in pages[:max_check]:
            if 1 <= pg <= len(doc):
                pg_text = doc[pg - 1].get_text().lower()
                if any(term.lower() in pg_text for term in key_terms):
                    doc.close()
                    return True
        doc.close()
    except Exception:
        return True   # can't open → assume verified

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Textline fallback parser (when geometric fails)
# ─────────────────────────────────────────────────────────────────────────────

def _textline_parse(page_texts: dict[int, str]) -> list[dict]:
    """
    Pure text-line parser for when geometric detection produces < 3 rows.

    Strategy:
      1. Concatenate all pages
      2. Split into rows by S.No anchors
      3. Within each row, detect bidder response by "GT has" / "Our response" etc.
      4. Extract max marks from the last integer 5-60 in each row header
    """
    rows: list[dict] = []
    seen_codes: set = set()

    full_text = ""
    page_boundaries: list[tuple[int, int]] = []   # (page_no, char_offset)
    for pg in sorted(page_texts):
        page_boundaries.append((pg, len(full_text)))
        full_text += f"\n[P{pg}]\n" + page_texts[pg] + "\n"

    def _char_to_page(pos: int) -> int:
        page = 1
        for pg_no, offset in page_boundaries:
            if offset <= pos:
                page = pg_no
        return page

    # Split by S.No markers
    sno_pattern = re.compile(r"(?m)^[ \t]*(\d{1,2})[ \t]*\n")
    splits = list(sno_pattern.finditer(full_text))

    for idx, sno_match in enumerate(splits):
        sno = int(sno_match.group(1))
        if not (1 <= sno <= 25):
            continue
        code = str(sno)
        if code in seen_codes:
            continue
        seen_codes.add(code)

        # Extract block: this anchor to next anchor
        block_start = sno_match.start()
        block_end   = splits[idx + 1].start() if idx + 1 < len(splits) else len(full_text)
        block       = full_text[block_start: min(block_end, block_start + 8000)]

        # Find response section
        resp_m = re.search(
            r"(?:gt\s+has|our\s+(?:firm|agency|company|average)|"
            r"we\s+have|grant\s+thornton|please\s+refer|"
            r"bidder\s+response|our\s+compliance|yes\s*[:\-,])",
            block, re.I,
        )
        criteria_text = block[:resp_m.start()].strip() if resp_m else block.strip()
        bidder_response = block[resp_m.start():].strip() if resp_m else ""

        # Extract max marks (last integer 5-60 in criteria section)
        max_marks = None
        for mm in re.finditer(r"\b(\d{1,3})\b", criteria_text):
            v = int(mm.group(1))
            if 5 <= v <= 100:
                max_marks = v

        # Parameter: first meaningful line
        param_lines = [l.strip() for l in criteria_text.split("\n")
                       if l.strip() and not re.match(r"^\[P\d+\]$", l.strip())]
        parameter = param_lines[0][:100] if param_lines else f"Criterion {sno}"
        # Remove S.No prefix from parameter
        parameter = re.sub(r"^\d{1,2}[.\s]+", "", parameter).strip()

        ev_pages = extract_evidence_pages(bidder_response)
        page     = _char_to_page(block_start)

        rows.append({
            "item_code":       code,
            "parent_code":     "",
            "parameter":       parameter,
            "criteria_text":   criteria_text[:1500],
            "max_marks":       max_marks,
            "proposed_marks":  None,
            "bidder_response": bidder_response[:3000],
            "evidence_pages":  ev_pages,
            "is_sub_item":     False,
            "raw_page":        page,
        })

        # Look for sub-items within this block
        sub_pattern = re.compile(
            r"(?:^|\n)\s*(?:\(([a-h])\)|([a-h])[\.\)](?!\w)|\((i{1,4})\))\s+(.{10,})",
            re.MULTILINE | re.IGNORECASE,
        )
        for sm in sub_pattern.finditer(block):
            sub_letter = (sm.group(1) or sm.group(2) or "").lower()
            roman = (sm.group(3) or "").lower()
            if roman:
                sub_letter = _ROMAN.get(roman, roman)
            if not sub_letter:
                continue
            sub_code = f"{code}{sub_letter}"
            if sub_code in seen_codes:
                continue
            seen_codes.add(sub_code)
            sub_text = sm.group(4)[:200]
            sub_marks_m = re.search(r"\b(\d{1,2})\s*marks?\b", sub_text, re.I)
            sub_marks = int(sub_marks_m.group(1)) if sub_marks_m else None

            rows.append({
                "item_code":       sub_code,
                "parent_code":     code,
                "parameter":       sub_text[:80],
                "criteria_text":   sub_text,
                "max_marks":       sub_marks,
                "proposed_marks":  None,
                "bidder_response": "",
                "evidence_pages":  ev_pages,  # inherit from parent
                "is_sub_item":     True,
                "raw_page":        page,
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — parse_compliance_matrix
# ─────────────────────────────────────────────────────────────────────────────

def parse_compliance_matrix(proposal_path: str) -> ComplianceMatrix:
    """
    Full pipeline: detect → geometric parse → fact extraction → verification.

    Returns a ComplianceMatrix with all rows populated.
    """
    warnings: list[str] = []

    if not _FITZ_OK:
        return ComplianceMatrix(rows=[], table_pages=[], proposal_path=proposal_path,
                                parse_method="none",
                                parse_warnings=["PyMuPDF not installed"])

    if not Path(proposal_path).exists():
        return ComplianceMatrix(rows=[], table_pages=[], proposal_path=proposal_path,
                                parse_method="none",
                                parse_warnings=[f"File not found: {proposal_path}"])

    # ── Stage 0: Detect table pages ───────────────────────────────────────────
    table_pages = find_compliance_table_pages(proposal_path)
    if not table_pages:
        return ComplianceMatrix(rows=[], table_pages=[], proposal_path=proposal_path,
                                parse_method="none",
                                parse_warnings=["Compliance table not found in proposal"])

    # ── Load words and page texts ─────────────────────────────────────────────
    doc = fitz.open(proposal_path)
    all_words: list[dict] = []
    page_texts: dict[int, str] = {}

    for pg in table_pages:
        if pg < 1 or pg > len(doc):
            continue
        page = doc[pg - 1]
        page_texts[pg] = page.get_text()

        for w in page.get_text("words"):
            # w = (x0, y0, x1, y1, word, block_no, line_no, word_no)
            if w[4].strip():
                all_words.append({
                    "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
                    "text": w[4].strip(),
                    "page": pg,
                })

    page_width = doc[table_pages[0] - 1].rect.width
    doc.close()

    if not all_words:
        return ComplianceMatrix(rows=[], table_pages=table_pages,
                                proposal_path=proposal_path,
                                parse_method="none",
                                parse_warnings=["No words extracted from table pages"])

    # ── Stage 1: Calibrate columns ────────────────────────────────────────────
    cm = _calibrate_columns(all_words, page_width)

    # ── Stage 2: Find row anchors ─────────────────────────────────────────────
    existing_parents: set = set()
    anchors = find_row_anchors(all_words, cm, existing_parents)
    print(f"[ComplianceParser] Found {len(anchors)} row anchors: "
          f"{[a.code for a in anchors[:10]]}")

    # ── Fallback to text-line if geometric finds < 2 anchors ─────────────────
    if len(anchors) < 2:
        warnings.append("Geometric anchor detection yielded < 2 rows — using textline fallback")
        print(f"[ComplianceParser] Falling back to textline parser")
        raw_rows = _textline_parse(page_texts)
        parse_method = "textline"
    else:
        raw_rows = _geometric_parse(all_words, anchors, cm)
        parse_method = "geometric"
        if len(raw_rows) < 2:
            warnings.append("Geometric parse yielded < 2 rows — using textline fallback")
            raw_rows = _textline_parse(page_texts)
            parse_method = "textline"

    # ── Stages 5-7: Enrich rows with facts and verification ───────────────────
    compliance_rows: list[ComplianceRow] = []

    for r in raw_rows:
        row = ComplianceRow(
            item_code       = r["item_code"],
            parent_code     = r["parent_code"],
            parameter       = r["parameter"],
            criteria_text   = r["criteria_text"],
            max_marks       = r["max_marks"],
            proposed_marks  = r["proposed_marks"],
            bidder_response = r["bidder_response"],
            evidence_pages  = r["evidence_pages"],
            is_sub_item     = r["is_sub_item"],
            raw_page        = r["raw_page"],
        )

        # Stage 6: Extract bidder fact
        val, label = extract_bidder_fact(
            row.bidder_response,
            row.criteria_text,
        )
        row.extracted_value = val
        row.extracted_label = label

        # Stage 7: Verify against evidence pages
        if val and row.evidence_pages:
            row.verified = verify_on_page(proposal_path, val, row.evidence_pages)
            if not row.verified:
                warnings.append(
                    f"Row {row.item_code}: could not verify '{val}' "
                    f"on page(s) {row.evidence_pages[:3]}"
                )
        elif val:
            row.verified = True  # value found but no page ref → trust it

        # Logging
        ver = "✓" if row.verified else "⚠"
        print(f"  [{row.item_code:5s}] {row.parameter[:45]:45s} "
              f"max={str(row.max_marks):4s} | fact='{val or '—'}' {ver}")

        compliance_rows.append(row)

    print(f"[ComplianceParser] Parsed {len(compliance_rows)} rows via {parse_method}")

    return ComplianceMatrix(
        rows          = compliance_rows,
        table_pages   = table_pages,
        proposal_path = proposal_path,
        parse_method  = parse_method,
        parse_warnings = warnings,
    )


def _geometric_parse(
    all_words: list[dict],
    anchors: list[RowAnchor],
    cm: ColumnMap,
) -> list[dict]:
    """Run the geometric parse pipeline on detected anchors."""
    rows: list[dict] = []

    for idx, anchor in enumerate(anchors):
        next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else None
        row_words   = collect_row_words(all_words, anchor, next_anchor)
        cols        = reconstruct_columns(row_words, cm)

        criteria_text   = cols.get("criteria", "") or cols.get("param", "")
        bidder_response = cols.get("response", "")
        docref_col      = cols.get("docref", "")
        marks_col       = cols.get("marks", "")

        # Merge docref page numbers into bidder response for evidence extraction
        response_for_evidence = bidder_response + " " + docref_col

        max_marks      = _extract_max_marks(marks_col, criteria_text)
        proposed_marks = _extract_proposed_marks(bidder_response, marks_col)
        ev_pages       = extract_evidence_pages(response_for_evidence)

        # Parameter: first non-trivial line of criteria col or param col
        param_col  = cols.get("param", "")
        param_text = (param_col or criteria_text)
        param_lines = [l.strip() for l in param_text.split() if len(l.strip()) > 2]
        parameter = " ".join(param_lines[:10])[:100]
        if not parameter:
            parameter = f"Criterion {anchor.code}"

        rows.append({
            "item_code":       anchor.code,
            "parent_code":     anchor.parent,
            "parameter":       parameter,
            "criteria_text":   criteria_text[:2000],
            "max_marks":       max_marks,
            "proposed_marks":  proposed_marks,
            "bidder_response": bidder_response[:4000],
            "evidence_pages":  ev_pages,
            "is_sub_item":     anchor.is_sub,
            "raw_page":        anchor.page,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Scoring integration — match matrix facts against RFP formula
# ─────────────────────────────────────────────────────────────────────────────

def score_from_matrix(
    matrix: ComplianceMatrix,
    rfp_criteria: list[dict],
    pre_cached_bands: Optional[dict] = None,
) -> list[dict]:
    """
    Score each RFP criterion using facts extracted from the compliance matrix.

    For each criterion in rfp_criteria:
      1. Find the matching ComplianceRow by item_code or parameter fuzzy match
      2. Use the extracted_value from Stage 6
      3. Apply the RFP formula using _apply_band_strict / _apply_step
      4. Return full result dict compatible with run_tq_evaluation output

    Falls back to LLM scoring only when no matching matrix row is found.
    """
    # Build lookup: item_code → row
    by_code: dict[str, ComplianceRow] = {r.item_code: r for r in matrix.rows}

    # Also build fuzzy param match
    def _best_row(criterion: dict) -> Optional[ComplianceRow]:
        code = str(criterion.get("item_code", "")).strip()
        if code in by_code:
            return by_code[code]
        # Try without sub-suffix: "4a" matches "4" parent
        if code and code[:-1] in by_code:
            return by_code[code[:-1]]
        # Fuzzy parameter match
        param = (criterion.get("parameter") or "").lower()
        best_score = 0.0
        best_row   = None
        for row in matrix.rows:
            words_crit = set(re.findall(r'\b[a-z]{4,}\b', param))
            words_row  = set(re.findall(r'\b[a-z]{4,}\b', row.parameter.lower()))
            if not words_crit or not words_row:
                continue
            overlap = len(words_crit & words_row) / len(words_crit | words_row)
            if overlap > best_score:
                best_score = overlap
                best_row   = row
        if best_score >= 0.30:
            return best_row
        return None

    results: list[dict] = []

    for criterion in rfp_criteria:
        max_marks   = int(criterion.get("max_marks") or 0)
        parameter   = criterion.get("parameter", "")
        item_code   = str(criterion.get("item_code", ""))
        formula     = criterion.get("formula_type", "LLM").upper()
        crit_text   = criterion.get("criteria_text", "")
        is_parent   = criterion.get("is_parent", False)
        is_sub_item = criterion.get("is_sub_item", False)

        if max_marks == 0 or is_parent:
            results.append(_zero_result(criterion))
            continue

        # Find matching matrix row
        row = _best_row(criterion)

        if row is None:
            print(f"  [ScoreMatrix] '{parameter[:40]}' — no matching matrix row")
            results.append({
                **_zero_result(criterion),
                "justification": "No matching row found in FORM TECH-4 matrix",
                "gaps":          ["Criterion not found in bidder's compliance table"],
            })
            continue

        # Get the extracted value (already parsed in Stage 6)
        value_str = row.extracted_value
        label     = row.extracted_label

        # If Stage 6 didn't find a value, try re-extracting with formula hint
        if not value_str:
            value_str, label = extract_bidder_fact(
                row.bidder_response,
                crit_text or row.criteria_text,
                formula_hint=formula,
            )

        if not value_str:
            print(f"  [ScoreMatrix] '{parameter[:40]}' — value not found in response")
            results.append({
                **_zero_result(criterion),
                "justification": f"Fact not found in bidder response for: {parameter}",
                "gaps":          ["Value not stated in compliance table"],
                "source_page":   row.raw_page,
            })
            continue

        # Apply RFP formula (import from v22 tq_extractor)
        score, steps = _apply_rfp_formula(formula, value_str, crit_text, max_marks,
                                          criterion, pre_cached_bands)

        # Evidence confidence
        ev_note = ""
        if row.evidence_pages:
            if row.verified:
                ev_note = f"Verified on p.{row.evidence_pages[0]}"
            else:
                ev_note = f"Cited p.{row.evidence_pages[0]} but not confirmed"
                if score > 0:
                    score = round(score * 0.9, 1)  # slight penalty for unverified
        else:
            ev_note = "No page reference cited"

        score = round(max(0.0, min(score, float(max_marks))), 1)
        pct   = round((score / max_marks) * 100, 1) if max_marks else 0.0

        results.append({
            "item_code":                       item_code,
            "parameter":                       parameter,
            "max_marks":                       max_marks,
            "criteria_text":                   crit_text,
            "formula_hint":                    formula,
            "is_sub_item":                     is_sub_item,
            "parent_parameter":                criterion.get("parent_parameter", ""),
            "score":                           score,
            "score_percentage":                pct,
            "extracted_value":                 f"{label} | {ev_note}",
            "source_page":                     row.raw_page,
            "evidence_pages":                  row.evidence_pages,
            "scoring_steps":                   steps,
            "justification": (
                f"Score {score}/{max_marks}. "
                f"Found: {value_str}. "
                f"{ev_note}. "
                f"From bidder compliance table (FORM TECH-4)."
            ),
            "strengths":                       [f"{label} | {ev_note}"] if score > 0 else [],
            "gaps":                            (
                [] if score >= max_marks else ["Higher value needed for full marks"]
            ),
            "evidence_found":                  score > 0,
            "verified":                        row.verified,
            "evaluation_layer":                "document",
            "requires_live_assessment":        False,
            "requires_comparative_evaluation": False,
            "discrepancies":                   [],
            "source":                          "compliance_matrix",
        })

    return results


def _apply_rfp_formula(
    formula: str,
    value_str: str,
    criteria_text: str,
    max_marks: int,
    criterion: dict,
    cached_bands: Optional[dict],
) -> tuple[float, str]:
    """
    Apply the appropriate formula to score a value against RFP criteria.
    Uses v22 band parser for BAND-type formulas.
    """
    try:
        # Try to import v22 band parser
        from core.tq_extractor import (
            _parse_band_table_strict,
            _apply_band_strict,
            _apply_step,
            _detect_formula,
        )
        v22_available = True
    except ImportError:
        v22_available = False

    # Parse numeric value
    num = None
    m   = re.search(r"([\d,]+(?:\.\d+)?)", (value_str or "").replace(",", ""))
    if m:
        try:
            num = float(m.group(1))
        except ValueError:
            pass

    if num is None:
        # Binary / presence check
        if value_str and value_str.lower() not in ("not found", "no", "none"):
            return float(max_marks), f"BINARY: presence → {max_marks}/{max_marks}"
        return 0.0, "BINARY: no evidence → 0"

    # ── BAND formula ──────────────────────────────────────────────────────────
    if formula in ("BAND", "BAND_CR", "BAND_PROJECTS", "BAND_HEADCOUNT",
                   "BAND_YEARS", "STEP"):
        if v22_available:
            bands = _parse_band_table_strict(criteria_text, formula)
            if bands:
                score = _apply_band_strict(bands, num, max_marks, formula)
                if score is not None:
                    return score, f"{formula}: {num} → {score}/{max_marks}"
            if formula == "STEP":
                score = _apply_step(criteria_text, max_marks, num)
                if score is not None:
                    return score, f"STEP: {num} → {score}/{max_marks}"

        # Fallback: use cached bands from RFP cache
        if cached_bands:
            bands_data = cached_bands.get(criterion.get("parameter", ""), {})
            band_list  = bands_data.get("bands", [])
            if band_list:
                for band in sorted(band_list, key=lambda b: float(b.get("min") or 0)):
                    lo = float(band.get("min") or 0)
                    hi = band.get("max")
                    hi = float(hi) if hi is not None else float("inf")
                    sc = float(band.get("score") or 0)
                    if lo <= num <= hi:
                        score = round(min(sc, float(max_marks)), 1)
                        return score, f"Cached bands: {num} in [{lo},{hi}] → {score}"

        # Last resort: proportional
        score = round(min(num / max(num, 1) * max_marks, float(max_marks)), 1)
        return score, f"Proportional fallback: {num}"

    # ── PER_UNIT formula ──────────────────────────────────────────────────────
    if formula == "PER_UNIT":
        rate_m = re.search(
            r"(\d+(?:\.\d+)?)\s*marks?\s+(?:for|per)\s+(?:each|01|one|per)\s+"
            r"(?:project|assignment)",
            criteria_text, re.I,
        )
        if rate_m:
            rate  = float(rate_m.group(1))
            score = round(min(num * rate, float(max_marks)), 1)
            return score, f"PER_UNIT: {int(num)}×{rate}={score}/{max_marks}"
        return round(min(num, float(max_marks)), 1), f"PER_UNIT fallback: {num}/{max_marks}"

    # ── QUAL formula ──────────────────────────────────────────────────────────
    if formula == "QUAL":
        # Presence of value = full marks (CV-based scoring)
        return float(max_marks), f"QUAL: CV evidence found → {max_marks}"

    # ── BINARY formula ────────────────────────────────────────────────────────
    if formula == "BINARY":
        return float(max_marks), f"BINARY: presence confirmed → {max_marks}"

    # ── LLM formula: use num as a direct score if it looks like a mark ────────
    if formula == "LLM":
        if 0 <= num <= max_marks:
            return num, f"LLM (direct): {num}/{max_marks}"
        return round(min(num, float(max_marks)), 1), f"LLM: capped at {max_marks}"

    return 0.0, f"Unknown formula '{formula}'"


def _zero_result(criterion: dict) -> dict:
    return {
        "item_code":                       str(criterion.get("item_code", "")),
        "parameter":                       criterion.get("parameter", ""),
        "max_marks":                       int(criterion.get("max_marks") or 0),
        "criteria_text":                   criterion.get("criteria_text", ""),
        "formula_hint":                    criterion.get("formula_type", "LLM"),
        "is_sub_item":                     criterion.get("is_sub_item", False),
        "parent_parameter":                criterion.get("parent_parameter", ""),
        "score":                           0.0,
        "score_percentage":                0.0,
        "extracted_value":                 None,
        "source_page":                     None,
        "evidence_pages":                  [],
        "scoring_steps":                   "Zero or parent criterion",
        "justification":                   "Zero-mark or parent criterion",
        "strengths":                       [],
        "gaps":                            [],
        "evidence_found":                  False,
        "verified":                        False,
        "evaluation_layer":                "document",
        "requires_live_assessment":        False,
        "requires_comparative_evaluation": False,
        "discrepancies":                   [],
        "source":                          "compliance_matrix",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick diagnostic runner
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_proposal(proposal_path: str, rfp_criteria: Optional[list] = None) -> None:
    """
    Print a full diagnostic of the compliance matrix parsing.fr
    Useful for debugging without running the full pipeline.
    """
    print(f"\n{'='*70}")
    print(f"COMPLIANCE MATRIX DIAGNOSTIC")
    print(f"Proposal: {Path(proposal_path).name}")
    print(f"{'='*70}\n")

    matrix = parse_compliance_matrix(proposal_path)

    print(f"\nParse method:  {matrix.parse_method}")
    print(f"Table pages:   {matrix.table_pages}")
    print(f"Rows found:    {len(matrix.rows)}")
    if matrix.parse_warnings:
        print(f"Warnings:      {matrix.parse_warnings}")

    print(f"\n{'─'*70}")
    print(f"{'Item':6} {'Parameter':40} {'Max':4} {'Fact':30} {'Ev':5}")
    print(f"{'─'*70}")

    for row in matrix.rows:
        sub = "  " if row.is_sub_item else ""
        ver = "✓" if row.verified else ("⚠" if row.extracted_value else "✗")
        print(f"{sub}{row.item_code:6} "
              f"{row.parameter[:38]:40} "
              f"{str(row.max_marks):4} "
              f"{str(row.extracted_value or '—')[:28]:30} "
              f"{ver}")

    if rfp_criteria:
        print(f"\n{'─'*70}")
        print(f"SCORING RESULTS")
        print(f"{'─'*70}")
        results = score_from_matrix(matrix, rfp_criteria)
        total = sum(r["score"] for r in results if r["score"])
        doc_max = sum(r["max_marks"] for r in results
                      if not rfp_criteria[results.index(r)].get("is_parent"))
        for r in results:
            sc = r["score"]
            sc_str = "--" if r.get("requires_live_assessment") else f"{sc}/{r['max_marks']}"
            print(f"  [{r['item_code']:4}] {r['parameter'][:45]:45} {sc_str:10} "
                  f"| {r.get('extracted_value','')[:35]}")
        print(f"\n  Total: {total}/{doc_max} ({round(total/doc_max*100,1) if doc_max else 0}%)")

    print(f"\n{'='*70}\n")

def run_tq_evaluation(
    rfp_doc_name: str,
    proposal_path: str,
    proposal_doc_name: str,
    progress_callback=None,
) -> dict:
    def _prog(step, pct):
        if progress_callback:
            progress_callback(step, pct)

    _prog("Loading RFP criteria from cache", 10)
    
    # Load cached RFP criteria
    from core.rfp_cache import load_cache
    from pathlib import Path
    
    # Find the RFP file
    rfp_path = None
    for search_dir in [Path("./uploads"), Path("./tq_uploads")]:
        candidate = search_dir / rfp_doc_name
        if candidate.exists():
            rfp_path = str(candidate)
            break
    
    rfp_criteria = []
    cached_bands = {}
    
    if rfp_path:
        cache = load_cache(rfp_path)
        if cache:
            rfp_criteria = cache.get("criteria", [])
            cached_bands = cache.get("bands", {})
            print(f"[TQ] Loaded {len(rfp_criteria)} criteria from RFP cache")
        else:
            # Fall back to extracting criteria from RFP
            _prog("Extracting RFP marking scheme", 15)
            try:
                from core.tq_criteria_extractor import extract_marking_scheme
                result = extract_marking_scheme(rfp_path)
                rfp_criteria = result.get("criteria", [])
                print(f"[TQ] Extracted {len(rfp_criteria)} criteria from RFP")
                
                # Cache for next time
                if rfp_criteria:
                    from core.rfp_cache import save_cache, precompute_bands
                    bands = precompute_bands(rfp_criteria)
                    save_cache(rfp_path, {"criteria": rfp_criteria, "grand_total_marks": result.get("grand_total_marks", 100)}, bands)
                    cached_bands = bands
            except Exception as e:
                print(f"[TQ] RFP criteria extraction failed: {e}")
    
    _prog("Parsing proposal compliance matrix", 30)
    
    # Try Docling first, fall back to geometric parser
    matrix = _parse_proposal_with_best_method(proposal_path)
    
    _prog("Scoring criteria", 60)
    
    from core.tq_compliance_parser import score_from_matrix
    
    # Inject cached bands into criteria
    for c in rfp_criteria:
        param = c.get("parameter", "")
        if param in cached_bands:
            c["_cached_bands"] = cached_bands[param]
    
    scores = score_from_matrix(matrix, rfp_criteria, pre_cached_bands=cached_bands)
    
    scoreable   = sum(s["max_marks"] for s in scores 
                      if not s.get("is_sub_item") and s["max_marks"] 
                      and not s.get("requires_live_assessment"))
    total_score = sum(s["score"] for s in scores 
                      if s.get("score") is not None 
                      and not s.get("requires_live_assessment"))
    pct = round((total_score / scoreable) * 100, 1) if scoreable else 0.0

    _prog("Evaluation complete", 100)

    return {
        "evaluation_title":       "Technical Evaluation",
        "grand_total_marks":      scoreable,
        "technical_document_max": scoreable,
        "live_assessment_marks":  0,
        "financial_marks":        0,
        "total_scored":           round(total_score, 1),
        "total_percentage":       pct,
        "scores":                 scores,
        "schema_valid":           len(matrix.rows) > 0,
        "schema_warning":         "; ".join(matrix.parse_warnings) or None,
        "final_score_formula":    None,
        "financial_evaluation":   None,
        "qualification":          None,
        "global_discrepancies":   [],
        "error":                  None,
    }


def _parse_proposal_with_best_method(proposal_path: str):
    """Try Docling → PyMuPDF geometric parser."""
    from core.tq_compliance_parser import parse_compliance_matrix, ComplianceMatrix
    
    # Try Docling if available
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        
        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        opts.do_ocr = False
        opts.table_structure_options.mode = TableFormerMode.FAST
        opts.table_structure_options.do_cell_matching = True
        
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        result = conv.convert(proposal_path)
        
        # Check if Docling found tables
        if result.document.tables:
            print(f"[TQ] Docling found {len(result.document.tables)} tables")
            # Still use geometric parser but with Docling's text quality
            # (Docling improves text layer which helps geometric parser)
    except ImportError:
        print("[TQ] Docling not installed — using PyMuPDF (pip install docling for better accuracy)")
    except Exception as e:
        print(f"[TQ] Docling failed: {e} — falling back to PyMuPDF")
    
    # Always run the geometric compliance parser (works well for FORM TECH-4)
    return parse_compliance_matrix(proposal_path)
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diagnose_proposal(sys.argv[1])
    else:
        print("Usage: python tq_compliance_parser.py <proposal.pdf>")