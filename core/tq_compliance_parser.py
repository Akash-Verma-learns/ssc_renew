"""
core/tq_compliance_parser.py  — v2  (OOM-safe, page-targeted Docling)
=====================================================================

FIXES vs v1
───────────
FIX 1 — Docling OOM crash
  v1 called Docling on the ENTIRE proposal PDF (852 pages → std::bad_alloc).
  v2 detects compliance table pages FIRST, then extracts ONLY those pages
  into a temp PDF before handing to Docling. For a 15-page table inside an
  852-page proposal, Docling sees 15 pages, not 852.

FIX 2 — Docling disabled by default for compliance parsing
  The FORM TECH-4 table is a standard multi-column grid that PyMuPDF's
  geometric parser handles well. Docling's TableFormer adds latency and
  memory cost without benefit here. Docling is now opt-in via
  DOCLING_TABLE_PARSE=1 env var, and only fires if the geometric parser
  yields < 2 rows.

FIX 3 — run_tq_evaluation moved to routes adapter
  The function in v1 that called `_parse_proposal_with_best_method` is
  replaced with a clean adapter that the routes.py background task calls.
  All Docling logic is behind a safe try/except that falls back to PyMuPDF.

FIX 4 — Temp-PDF extraction for targeted parsing
  _extract_pages_to_temp_pdf() extracts only the detected table pages to a
  temp file. Any downstream tool (Docling, pdfplumber, etc.) works on that.
"""

from __future__ import annotations

import os
import re
import tempfile
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
    item_code:        str
    parent_code:      str
    parameter:        str
    criteria_text:    str
    max_marks:        Optional[int]
    proposed_marks:   Optional[int]
    bidder_response:  str
    evidence_pages:   list
    is_sub_item:      bool = False
    raw_page:         int  = 0
    extracted_value:  Optional[str] = None
    extracted_label:  str = ""
    verified:         bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ComplianceMatrix:
    rows:           list
    table_pages:    list
    proposal_path:  str
    parse_method:   str = "geometric"
    parse_warnings: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Multi-signal compliance table page detection
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_SIGNALS = re.compile(
    r"""(
        form\s*tech[\s\-]*4                                                   |
        technical\s+bid\s+evaluation\s+criteria\s+and\s+(our\s+)?compliance   |
        technical\s+evaluation\s+criteria\s+and\s+(our\s+)?response           |
        evaluation\s+criteria\s+and\s+compliance\s+statement                  |
        our\s+compliance\s+(to|with|against)\s+(the\s+)?criteria              |
        self[\s\-]assessment\s+(table|criteria|form)                          |
        sl\.?\s*no\.?.{0,60}criteria.{0,60}(proposed|gt|bidder)\s+marks      |
        s\.?\s*no\.?.{0,60}qualification.{0,60}marks?\s+awarded              |
        s\.?\s*no\.?.{0,60}parameter.{0,60}gt\s+marks
    )""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

_COL_HEADERS = re.compile(
    r"\b(proposed\s+marks?|gt\s+marks?|marks?\s+awarded|bidder\s+marks?|"
    r"supporting\s+doc|reference\s+document|page\s+(no|reference)|"
    r"our\s+response|bidder\s+response|compliance\s+details?|remarks?)\b",
    re.IGNORECASE,
)

_MARKS_RE   = re.compile(r"\b(\d{1,2})\s*marks?\b", re.I)
_PAGE_REF_RE = re.compile(
    r"(?:page|pg)\.?\s*(?:no\.?|nos?\.?)?\s*(\d+(?:\s*[-–to]+\s*\d+)?)",
    re.IGNORECASE,
)


def _score_page_for_table(page_text: str, page_lower: str) -> float:
    score = 0.0
    if _HEADER_SIGNALS.search(page_text):   score += 15.0
    score += len(_COL_HEADERS.findall(page_text)) * 2.5
    score += min(len(_MARKS_RE.findall(page_text)) * 0.8, 6.0)
    score += min(len(_PAGE_REF_RE.findall(page_text)) * 0.6, 4.0)
    if re.search(r"grant\s*thornton|gtbl|gt\s+has|our\s+firm|we\s+have", page_lower):
        score += 3.0
    if re.search(r"(?m)^\s*\d{1,2}\s*\n", page_text):
        score += 2.0
    return score


def find_compliance_table_pages(
    proposal_path: str,
    max_search_pages: int = 120,
    min_page_score: float = 4.0,
) -> list:
    """
    Find PDF pages forming the FORM TECH-4 compliance matrix.
    Searches only the first max_search_pages pages (avoids scanning 800+ pages).
    Returns sorted list of 1-based page numbers.
    """
    if not _FITZ_OK or not Path(proposal_path).exists():
        return []

    doc = fitz.open(proposal_path)
    n   = min(len(doc), max_search_pages)

    page_scores = []
    for i in range(n):
        txt   = doc[i].get_text()
        lower = txt.lower()
        s     = _score_page_for_table(txt, lower)
        page_scores.append((s, i + 1))
    doc.close()

    if not any(s > min_page_score for s, _ in page_scores):
        print("[ComplianceParser] No table pages detected above threshold")
        return []

    strong_starts = [pg for s, pg in page_scores if s >= 12.0]
    if not strong_starts:
        sorted_by_score = sorted(page_scores, key=lambda x: -x[0])
        strong_starts = [sorted_by_score[0][1]]

    start_pg = min(strong_starts)
    print(f"[ComplianceParser] Table starts at page {start_pg} "
          f"(score={page_scores[start_pg-1][0]:.1f})")

    table_pages = [start_pg]
    doc2 = fitz.open(proposal_path)
    for pg in range(start_pg + 1, n + 1):
        s = page_scores[pg - 1][0]
        txt_lower = ""
        if pg <= len(doc2):
            txt_lower = doc2[pg - 1].get_text().lower()

        is_break = bool(re.search(
            r"(annex[u]?r[e]?\s+[a-z]|appendix\s+[a-z]|"
            r"financial\s+proposal|commercial\s+bid|"
            r"chapter\s+[ivxlc]+)",
            txt_lower,
        ))
        if is_break and s < 3.0:
            print(f"[ComplianceParser] Table ends before page {pg} (section break)")
            break
        if s < min_page_score and len(table_pages) > 2:
            if page_scores[table_pages[-1] - 1][0] < min_page_score:
                print(f"[ComplianceParser] Table ends before page {pg} (low score)")
                break
        table_pages.append(pg)
    doc2.close()

    table_pages = table_pages[:35]   # safety cap
    print(f"[ComplianceParser] Table pages: {table_pages}")
    return table_pages


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1+2 — Safe Docling: extract only table pages to temp PDF first
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pages_to_temp_pdf(src_path: str, page_nos: list) -> Optional[str]:
    """
    Extract the given 1-based page numbers to a temporary PDF.
    Returns temp file path or None on failure.
    """
    if not _FITZ_OK or not page_nos:
        return None
    try:
        src = fitz.open(src_path)
        dst = fitz.open()
        for pg in page_nos:
            idx = pg - 1
            if 0 <= idx < len(src):
                dst.insert_pdf(src, from_page=idx, to_page=idx)
        src.close()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        dst.save(tmp.name)
        dst.close()
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"[ComplianceParser] Temp PDF extraction failed: {e}")
        return None


def _try_docling_on_pages(src_path: str, page_nos: list) -> Optional[list]:
    """
    FIX 1: Run Docling ONLY on the extracted table pages (not the full PDF).
    Returns list of (table_markdown, page_no) tuples or None if unavailable/OOM.
    """
    if not os.getenv("DOCLING_TABLE_PARSE", ""):
        return None   # opt-in only

    tmp_path = _extract_pages_to_temp_pdf(src_path, page_nos)
    if not tmp_path:
        return None

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        opts.do_ocr = False
        opts.table_structure_options.mode = TableFormerMode.FAST
        opts.table_structure_options.do_cell_matching = True

        conv   = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        result = conv.convert(tmp_path)
        tables = []
        for tbl in result.document.tables:
            try:
                md = tbl.export_to_markdown(result.document)
                # Map local page back to original page
                local_pg = tbl.prov[0].page_no if tbl.prov else 1
                orig_pg  = page_nos[local_pg - 1] if local_pg <= len(page_nos) else page_nos[0]
                tables.append((md, orig_pg))
            except Exception:
                pass
        print(f"[ComplianceParser] Docling found {len(tables)} tables on {len(page_nos)} pages")
        return tables if tables else None

    except MemoryError:
        print("[ComplianceParser] Docling OOM on table pages — falling back to PyMuPDF")
        return None
    except Exception as e:
        print(f"[ComplianceParser] Docling failed: {e} — falling back to PyMuPDF")
        return None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Geometric column calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnMap:
    page_width:  float
    sno_lo:      float = 0.0
    sno_hi:      float = 0.0
    param_lo:    float = 0.0
    param_hi:    float = 0.0
    crit_lo:     float = 0.0
    crit_hi:     float = 0.0
    marks_lo:    float = 0.0
    marks_hi:    float = 0.0
    response_lo: float = 0.0
    response_hi: float = 0.0
    doc_ref_lo:  float = 0.0
    doc_ref_hi:  float = 0.0

    def classify_x(self, x: float) -> str:
        if self.sno_lo <= x <= self.sno_hi:           return "sno"
        if self.param_lo <= x <= self.param_hi:       return "param"
        if self.crit_lo <= x <= self.crit_hi:         return "criteria"
        if self.marks_lo <= x <= self.marks_hi:       return "marks"
        if self.doc_ref_lo <= x <= self.doc_ref_hi:   return "docref"
        if self.response_lo <= x <= self.response_hi: return "response"
        return "unknown"


def _calibrate_columns(words: list, page_width: float) -> ColumnMap:
    cm = ColumnMap(page_width=page_width)
    pw = page_width

    y_clusters: dict = defaultdict(list)
    for w in words:
        y_key = round(w["y0"] / 6) * 6
        y_clusters[y_key].append(w)

    header_row_words: list = []
    for y_key in sorted(y_clusters):
        row_words = y_clusters[y_key]
        row_text  = " ".join(w["text"] for w in row_words).lower()
        kw_count  = sum(1 for kw in [
            "s.no", "s. no", "no.", "criteria", "parameter", "marks",
            "response", "reference", "document", "support",
        ] if kw in row_text)
        if kw_count >= 3:
            header_row_words.extend(row_words)

    def _find_col_x(pattern: str, wds: list) -> Optional[float]:
        for w in wds:
            if re.search(pattern, w["text"], re.I):
                return (w["x0"] + w["x1"]) / 2
        return None

    sno_x      = _find_col_x(r"^S\.?$|^SL\.?$", header_row_words)
    crit_x     = _find_col_x(r"criteria|parameter|particulars|criterion", header_row_words)
    marks_x    = _find_col_x(r"max(imum)?|marks?|full\s+marks?", header_row_words)
    response_x = _find_col_x(r"response|compliance|bidder|gt\s+has|our\s+claim", header_row_words)
    docref_x   = _find_col_x(r"document|reference|annexure|support|page\s*(no|ref)", header_row_words)

    if sno_x is None:
        left_xs = [w["x0"] for w in words if w["x0"] < pw * 0.12]
        if left_xs:
            from collections import Counter
            rounded = [round(x / 4) * 4 for x in left_xs]
            sno_x = Counter(rounded).most_common(1)[0][0]
        else:
            sno_x = pw * 0.05

    if marks_x is None:
        right_zone_xs = [w["x0"] for w in words
                         if pw * 0.55 < w["x0"] < pw * 0.78
                         and re.match(r"^\d{1,2}$", w["text"])
                         and 5 <= int(w["text"]) <= 60]
        marks_x = (sorted(right_zone_xs)[len(right_zone_xs) // 2]
                   if right_zone_xs else pw * 0.65)

    if response_x is None: response_x = pw * 0.82
    if docref_x   is None: docref_x   = pw * 0.90
    if crit_x     is None: crit_x     = pw * 0.38

    cm.sno_lo    = max(0.0, sno_x - 15)
    cm.sno_hi    = sno_x + 30
    cm.param_lo  = cm.sno_hi
    cm.param_hi  = (crit_x - 20) if crit_x > cm.sno_hi + 40 else cm.sno_hi + 60
    cm.crit_lo   = cm.param_hi
    cm.crit_hi   = marks_x - 15 if marks_x > cm.crit_lo + 40 else pw * 0.60
    cm.marks_lo  = marks_x - 20
    cm.marks_hi  = marks_x + 40

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
        cm.response_lo = marks_x + 40
        cm.response_hi = pw
        cm.doc_ref_lo  = pw * 0.88
        cm.doc_ref_hi  = pw

    print(f"[ComplianceParser] Columns: sno=[{cm.sno_lo:.0f},{cm.sno_hi:.0f}] "
          f"crit=[{cm.crit_lo:.0f},{cm.crit_hi:.0f}] "
          f"marks=[{cm.marks_lo:.0f},{cm.marks_hi:.0f}] "
          f"resp=[{cm.response_lo:.0f},{cm.response_hi:.0f}]")
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2-3 — Row anchors + multi-page assembly
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RowAnchor:
    code:   str
    parent: str
    page:   int
    y0:     float
    is_sub: bool


_ROMAN = {"i": "a", "ii": "b", "iii": "c", "iv": "d", "v": "e",
          "vi": "f", "vii": "g", "viii": "h"}

_SUBLABEL_RE = re.compile(
    r"^\s*(?:"
    r"\(([a-h])\)|([a-h])[\.\)](?!\w)"
    r"|\((i{1,4}v?|vi{0,3})\)|(i{1,4}v?|vi{0,3})[\.\)](?!\w)"
    r")\s*$",
    re.IGNORECASE,
)


def _parse_anchor_code(text: str, parent: str) -> Optional[tuple]:
    text = text.strip()
    m = _SUBLABEL_RE.match(text)
    if not m: return None
    letter = (m.group(1) or m.group(2) or "").lower()
    roman  = (m.group(3) or m.group(4) or "").lower()
    if roman: letter = _ROMAN.get(roman, roman)
    if not letter: return None
    code = f"{parent}{letter}" if parent else letter
    return code, True


def find_row_anchors(words: list, cm: ColumnMap, existing_parents: set) -> list:
    anchors   = []
    seen_codes: set = set()

    for w in words:
        if not (cm.sno_lo <= w["x0"] <= cm.sno_hi): continue
        text = w["text"].rstrip(".")
        if not re.match(r"^\d{1,2}$", text): continue
        val = int(text)
        if not (1 <= val <= 25): continue
        if w["y0"] < 40: continue
        code = str(val)
        if code not in seen_codes:
            seen_codes.add(code)
            anchors.append(RowAnchor(code=code, parent="", page=w["page"],
                                     y0=w["y0"], is_sub=False))

    current_parent = ""
    for w in sorted(words, key=lambda x: (x["page"], x["y0"])):
        parent_anchor = next(
            (a for a in anchors if a.page == w["page"]
             and abs(a.y0 - w["y0"]) < 5 and not a.is_sub), None)
        if parent_anchor:
            current_parent = parent_anchor.code
            continue
        if not (cm.sno_lo - 5 <= w["x0"] <= cm.param_hi): continue
        if w["y0"] < 40: continue
        result = _parse_anchor_code(w["text"], current_parent)
        if result is None: continue
        code, is_sub = result
        if code not in seen_codes:
            seen_codes.add(code)
            anchors.append(RowAnchor(code=code, parent=current_parent,
                                     page=w["page"], y0=w["y0"], is_sub=True))

    anchors.sort(key=lambda a: (a.page, a.y0))
    return anchors


def collect_row_words(words: list, anchor: RowAnchor, next_anchor: Optional[RowAnchor]) -> list:
    row_words = []
    for w in words:
        if w["page"] < anchor.page: continue
        if next_anchor and w["page"] > next_anchor.page: break
        if w["page"] == anchor.page and w["y0"] < anchor.y0 - 3: continue
        if next_anchor and w["page"] == next_anchor.page and w["y0"] >= next_anchor.y0 - 3: break
        row_words.append(w)
    return row_words


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Column-aware text reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_columns(row_words: list, cm: ColumnMap) -> dict:
    col_lines: dict = defaultdict(lambda: defaultdict(list))
    for w in row_words:
        col = cm.classify_x(w["x0"])
        if col == "unknown":
            col = cm.classify_x((w["x0"] + w["x1"]) / 2)
        if col == "unknown":
            centres = {
                "sno":      (cm.sno_lo + cm.sno_hi) / 2,
                "param":    (cm.param_lo + cm.param_hi) / 2,
                "criteria": (cm.crit_lo + cm.crit_hi) / 2,
                "marks":    (cm.marks_lo + cm.marks_hi) / 2,
                "response": (cm.response_lo + cm.response_hi) / 2,
                "docref":   (cm.doc_ref_lo + cm.doc_ref_hi) / 2,
            }
            col = min(centres, key=lambda c: abs(centres[c] - w["x0"]))
        y_key = round(w["y0"] / 4) * 4
        col_lines[col][y_key].append(w)

    result: dict = {}
    for col, line_dict in col_lines.items():
        lines = []
        for y_key in sorted(line_dict):
            line_words = sorted(line_dict[y_key], key=lambda w: w["x0"])
            lines.append(" ".join(w["text"] for w in line_words).strip())
        result[col] = " ".join(l for l in lines if l).strip()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Marks extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_max_marks(marks_col_text: str, criteria_col_text: str) -> Optional[int]:
    for m in re.finditer(r"\b(\d{1,3})\b", marks_col_text or ""):
        val = int(m.group(1))
        if 1 <= val <= 100: return val
    m2 = re.search(r"max(?:imum)?\s*(?:of\s*)?(\d{1,3})\s*marks?", criteria_col_text, re.I)
    if m2:
        val = int(m2.group(1))
        if 1 <= val <= 100: return val
    for m3 in re.finditer(r"\b(\d{1,2})\s*marks?\b", criteria_col_text, re.I):
        val = int(m3.group(1))
        if 5 <= val <= 60: return val
    return None


def _extract_proposed_marks(response_col_text: str, marks_col_text: str) -> Optional[int]:
    for text in [response_col_text, marks_col_text]:
        m = re.search(r"(?:proposed|claimed|awarded|scored|marks?\s*[:\-=])?\s*(\d{1,2})", text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 60: return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Python-first fact extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_bidder_fact(
    bidder_response: str,
    criteria_text: str,
    formula_hint: str = "",
) -> tuple:
    """
    Extract numeric/qualifying fact from bidder's response.
    Returns (value_string, label_string) or (None, "Not found").
    """
    txt  = bidder_response or ""
    crit = (criteria_text or "").lower()

    # Guard: reject if response is purely scoring language
    if re.search(r"\b(marks?\s+for|marks?\s+per|maximum\s+marks?|scoring\s+criteria)\b", txt, re.I):
        non_scoring = re.sub(
            r"[^.]*\b(?:marks?\s+for|marks?\s+per|maximum\s+marks?)[^.]*\.", "", txt).strip()
        if len(non_scoring) < 40:
            return None, "Response appears to be scoring criteria, not bidder claim"

    # ── TURNOVER ──────────────────────────────────────────────────────────────
    if any(k in crit for k in ["turnover", "crore", "financial", "revenue"]):
        turnover_patterns = [
            (r"average\s+(?:annual\s+)?turnover\s+(?:of\s+)?(?:inr|rs\.?|₹)?\s*"
             r"([\d,]+(?:\.\d+)?)\s*(?:cr(?:ores?)?|lakh)?", "avg turnover"),
            (r"(?:inr|rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:cr(?:ores?)?|lakh)", "INR value"),
            (r"([\d,]+(?:\.\d+)?)\s*cr(?:ores?)?\b(?!\s*per\b)", "Crore value"),
            (r"(?:rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*cr\b", "Rs Cr"),
            (r"average\s+([\d,]+(?:\.\d+)?)", "average from table"),
        ]
        for pat, label in turnover_patterns:
            m = re.search(pat, txt, re.I)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if 0.1 <= val <= 100_000:
                        if "lakh" in m.group(0).lower():
                            val = round(val / 100, 2)
                        return f"{val} Cr", f"{val} Cr ({label})"
                except ValueError:
                    pass

    # ── PROJECT COUNT ─────────────────────────────────────────────────────────
    if any(k in crit for k in ["project", "assignment", "pmc", "pmu", "urban",
                                "billing", "work order", "experience"]):
        project_patterns = [
            (r"following\s+(\d+)\s+(?:such\s+)?(?:assign|project)", "following N"),
            (r"handled\s+(\d+)\s+(?:large|major)?\s*(?:scale\s+)?project", "handled N"),
            (r"(\d+)\s+(?:such\s+)?(?:project|assignment)s?\s+(?:have\s+)?(?:been\s+)?completed", "N completed"),
            (r"enclosed?\s+(?:details?\s+of\s+)?(\d+)\s+(?:such\s+)?(?:project|assign)", "enclosed N"),
            (r"(\d+)\s+(?:eligible|qualifying|relevant)\s+(?:project|assign)", "N eligible"),
        ]
        for pat, label in project_patterns:
            m = re.search(pat, txt, re.I)
            if m:
                count = int(m.group(1))
                if 1 <= count <= 50:
                    return str(count), f"{count} projects ({label})"
        # Fallback: count numbered list entries
        nums = re.findall(r"(?:^|\n)\s*(\d{1,2})\.\s+[A-Z]", txt, re.MULTILINE)
        if nums:
            count = max(int(n) for n in nums)
            if 1 <= count <= 50:
                return str(count), f"{count} projects (numbered list)"

    # ── PERSONNEL / HEADCOUNT ─────────────────────────────────────────────────
    if any(k in crit for k in ["manpower", "professional", "personnel", "employee",
                                "staff", "headcount"]):
        personnel_patterns = [
            (r"more\s+than\s+([\d,]+)\s+technically\s+qualified", "more than N technically qualified"),
            (r"([\d,]+)\+?\s+technically\s+qualified", "N technically qualified"),
            (r"more\s+than\s+([\d,]+)\s+(?:consulting|advisory|qualified)?\s*(?:staff|personnel|professional)",
             "more than N staff"),
            (r"([\d,]+)\+?\s+(?:consulting|advisory|qualified)\s*(?:staff|personnel|professional)",
             "N consulting staff"),
            (r"payroll\s+of\s+([\d,]+)", "payroll of N"),
            (r"deployed\s+([\d,]+)\s+(?:professionals?|resources?|experts?)", "deployed N"),
        ]
        for pat, label in personnel_patterns:
            m = re.search(pat, txt, re.I)
            if m:
                try:
                    val = int(m.group(1).replace(",", ""))
                    if val > 0: return str(val), f"{val} ({label})"
                except ValueError:
                    pass

    # ── EXPERIENCE YEARS ──────────────────────────────────────────────────────
    if any(k in crit for k in ["experience", "year", "since", "established"]):
        year_patterns = [
            (r"more\s+than\s+(\d+)\s+years?\s+(?:of\s+)?(?:relevant\s+)?experience", "more than N years"),
            (r"(\d+)\+\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience", "N+ years"),
            (r"(\d+)\s+years?\s+of\s+(?:relevant\s+)?(?:experience|expertise)", "N years"),
            (r"since\s+(20\d{2}|19\d{2})", "since YYYY"),
            (r"established\s+in\s+(20\d{2}|19\d{2})", "established YYYY"),
        ]
        for pat, label in year_patterns:
            m = re.search(pat, txt, re.I)
            if m:
                if label in ("since YYYY", "established YYYY"):
                    try:
                        yrs = 2024 - int(m.group(1))
                        return f"more than {yrs} years", f"{yrs} years ({m.group(1)})"
                    except ValueError:
                        pass
                else:
                    try:
                        yrs = int(m.group(1))
                        if 1 <= yrs <= 50: return f"{yrs} years", f"{yrs} years ({label})"
                    except ValueError:
                        pass

    # ── BINARY / PRESENCE ────────────────────────────────────────────────────
    for pat in [
        r"grant\s+thornton\s+has", r"gt\s+has\s+(?:extensive\s+)?experience",
        r"we\s+have\s+(?:extensive\s+)?experience",
        r"please\s+refer\s+(?:form|page|annex)", r"enclosed\s+(?:herewith|at)",
        r"methodology\s+is\s+(?:enclosed|provided|attached)",
        r"\byes\b", r"registered\s+(?:with|under)",
    ]:
        if re.search(pat, txt, re.I):
            return "Yes", "Presence confirmed (positive claim in response)"

    return None, "Not found in bidder response"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Evidence page extraction and spot-verification
# ─────────────────────────────────────────────────────────────────────────────

def extract_evidence_pages(bidder_response: str, docref_col: str = "") -> list:
    pages: set = set()
    combined = (bidder_response or "") + " " + (docref_col or "")
    for m in _PAGE_REF_RE.finditer(combined):
        ref = m.group(1)
        range_m = re.match(r"(\d+)\s*[-–to]+\s*(\d+)", ref)
        if range_m:
            lo = int(range_m.group(1))
            if 1 <= lo <= 1000: pages.add(lo)
        else:
            try:
                pg = int(ref.strip())
                if 1 <= pg <= 1000: pages.add(pg)
            except ValueError:
                pass
    return sorted(pages)


def verify_on_page(pdf_path: str, claimed_value: str, pages: list, max_check: int = 3) -> bool:
    if not pages or not claimed_value or not _FITZ_OK: return False
    nums = re.findall(r"\d[\d,.]*", claimed_value)
    key_terms = [n.replace(",", "") for n in nums[:3]] or [claimed_value[:20].lower()]
    try:
        doc = fitz.open(pdf_path)
        for pg in pages[:max_check]:
            if 1 <= pg <= len(doc):
                if any(t.lower() in doc[pg-1].get_text().lower() for t in key_terms):
                    doc.close(); return True
        doc.close()
    except Exception:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Textline fallback parser
# ─────────────────────────────────────────────────────────────────────────────

_ROMAN_MAP = {"i": "a", "ii": "b", "iii": "c", "iv": "d", "v": "e"}


def _textline_parse(page_texts: dict) -> list:
    rows       = []
    seen_codes: set = set()
    full_text  = ""
    page_boundaries = []

    for pg in sorted(page_texts):
        page_boundaries.append((pg, len(full_text)))
        full_text += f"\n[P{pg}]\n" + page_texts[pg] + "\n"

    def _char_to_page(pos: int) -> int:
        page = 1
        for pg_no, offset in page_boundaries:
            if offset <= pos: page = pg_no
        return page

    sno_pattern = re.compile(r"(?m)^[ \t]*(\d{1,2})[ \t]*\n")
    splits = list(sno_pattern.finditer(full_text))

    for idx, sno_match in enumerate(splits):
        sno = int(sno_match.group(1))
        if not (1 <= sno <= 25): continue
        code = str(sno)
        if code in seen_codes: continue
        seen_codes.add(code)

        block_start = sno_match.start()
        block_end   = splits[idx+1].start() if idx+1 < len(splits) else len(full_text)
        block       = full_text[block_start:min(block_end, block_start+8000)]

        resp_m = re.search(
            r"(?:gt\s+has|our\s+(?:firm|agency|company|average)|"
            r"we\s+have|grant\s+thornton|please\s+refer|"
            r"bidder\s+response|our\s+compliance|yes\s*[:\-,])",
            block, re.I,
        )
        criteria_text   = block[:resp_m.start()].strip() if resp_m else block.strip()
        bidder_response = block[resp_m.start():].strip() if resp_m else ""

        max_marks = None
        for mm in re.finditer(r"\b(\d{1,3})\b", criteria_text):
            v = int(mm.group(1))
            if 5 <= v <= 100: max_marks = v

        param_lines = [l.strip() for l in criteria_text.split("\n")
                       if l.strip() and not re.match(r"^\[P\d+\]$", l.strip())]
        parameter = re.sub(r"^\d{1,2}[.\s]+", "", param_lines[0]).strip()[:100] if param_lines else f"Criterion {sno}"
        ev_pages  = extract_evidence_pages(bidder_response)
        page      = _char_to_page(block_start)

        rows.append({
            "item_code": code, "parent_code": "", "parameter": parameter,
            "criteria_text": criteria_text[:1500], "max_marks": max_marks,
            "proposed_marks": None, "bidder_response": bidder_response[:3000],
            "evidence_pages": ev_pages, "is_sub_item": False, "raw_page": page,
        })

        sub_pattern = re.compile(
            r"(?:^|\n)\s*(?:\(([a-h])\)|([a-h])[\.\)](?!\w)|\((i{1,4})\))\s+(.{10,})",
            re.MULTILINE | re.IGNORECASE,
        )
        for sm in sub_pattern.finditer(block):
            sub_letter = (sm.group(1) or sm.group(2) or "").lower()
            roman = (sm.group(3) or "").lower()
            if roman: sub_letter = _ROMAN_MAP.get(roman, roman)
            if not sub_letter: continue
            sub_code = f"{code}{sub_letter}"
            if sub_code in seen_codes: continue
            seen_codes.add(sub_code)
            sub_text = sm.group(4)[:200]
            sub_marks_m = re.search(r"\b(\d{1,2})\s*marks?\b", sub_text, re.I)
            rows.append({
                "item_code": sub_code, "parent_code": code,
                "parameter": sub_text[:80], "criteria_text": sub_text,
                "max_marks": int(sub_marks_m.group(1)) if sub_marks_m else None,
                "proposed_marks": None, "bidder_response": "",
                "evidence_pages": ev_pages, "is_sub_item": True, "raw_page": page,
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — parse_compliance_matrix
# ─────────────────────────────────────────────────────────────────────────────

def parse_compliance_matrix(proposal_path: str) -> ComplianceMatrix:
    """
    Full pipeline: detect table pages → geometric parse → fact extraction → verification.
    Docling is only used on the detected table pages (FIX 1), not the whole PDF.
    """
    warnings: list = []

    if not _FITZ_OK:
        return ComplianceMatrix(rows=[], table_pages=[], proposal_path=proposal_path,
                                parse_method="none",
                                parse_warnings=["PyMuPDF not installed"])

    if not Path(proposal_path).exists():
        return ComplianceMatrix(rows=[], table_pages=[], proposal_path=proposal_path,
                                parse_method="none",
                                parse_warnings=[f"File not found: {proposal_path}"])

    # Stage 0: Detect table pages (FIX 1: only search first 120 pages)
    table_pages = find_compliance_table_pages(proposal_path)
    if not table_pages:
        return ComplianceMatrix(rows=[], table_pages=[], proposal_path=proposal_path,
                                parse_method="none",
                                parse_warnings=["Compliance table not found in proposal"])

    # Load words from table pages ONLY
    doc = fitz.open(proposal_path)
    all_words: list = []
    page_texts: dict = {}

    for pg in table_pages:
        if pg < 1 or pg > len(doc): continue
        page = doc[pg - 1]
        page_texts[pg] = page.get_text()
        for w in page.get_text("words"):
            if w[4].strip():
                all_words.append({
                    "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
                    "text": w[4].strip(), "page": pg,
                })

    page_width = doc[table_pages[0] - 1].rect.width
    doc.close()

    if not all_words:
        return ComplianceMatrix(rows=[], table_pages=table_pages,
                                proposal_path=proposal_path, parse_method="none",
                                parse_warnings=["No words extracted from table pages"])

    # Stage 1: Calibrate columns
    cm = _calibrate_columns(all_words, page_width)

    # Stage 2: Find row anchors
    anchors = find_row_anchors(all_words, cm, set())
    print(f"[ComplianceParser] Found {len(anchors)} row anchors: "
          f"{[a.code for a in anchors[:10]]}")

    # FIX 2: Docling only on table pages, only if opt-in AND geometric fails
    raw_rows   = []
    parse_method = "geometric"

    if len(anchors) >= 2:
        raw_rows = _geometric_parse(all_words, anchors, cm)
        if len(raw_rows) < 2:
            warnings.append("Geometric parse < 2 rows — trying textline")
            raw_rows     = _textline_parse(page_texts)
            parse_method = "textline"
    else:
        warnings.append("Geometric anchors < 2 — using textline fallback")
        raw_rows     = _textline_parse(page_texts)
        parse_method = "textline"

        # FIX 2: only try Docling if textline also fails AND env var is set
        if len(raw_rows) < 2 and os.getenv("DOCLING_TABLE_PARSE", ""):
            print("[ComplianceParser] Textline also failed — trying Docling on table pages only")
            docling_tables = _try_docling_on_pages(proposal_path, table_pages)
            if docling_tables:
                # Parse Docling markdown output
                for md, pg in docling_tables:
                    raw_rows.extend(_parse_markdown_table(md, pg))
                parse_method = "docling"

    # Stages 5-7: Enrich rows
    compliance_rows: list = []
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
        val, label = extract_bidder_fact(row.bidder_response, row.criteria_text)
        row.extracted_value = val
        row.extracted_label = label

        if val and row.evidence_pages:
            row.verified = verify_on_page(proposal_path, val, row.evidence_pages)
            if not row.verified:
                warnings.append(f"Row {row.item_code}: could not verify '{val}' "
                                f"on page(s) {row.evidence_pages[:3]}")
        elif val:
            row.verified = True

        ver = "✓" if row.verified else "⚠"
        print(f"  [{row.item_code:5s}] {row.parameter[:45]:45s} "
              f"max={str(row.max_marks):4s} | fact='{val or '—'}' {ver}")
        compliance_rows.append(row)

    print(f"[ComplianceParser] Parsed {len(compliance_rows)} rows via {parse_method}")
    return ComplianceMatrix(rows=compliance_rows, table_pages=table_pages,
                            proposal_path=proposal_path, parse_method=parse_method,
                            parse_warnings=warnings)


def _geometric_parse(all_words: list, anchors: list, cm: ColumnMap) -> list:
    rows = []
    for idx, anchor in enumerate(anchors):
        next_anchor = anchors[idx+1] if idx+1 < len(anchors) else None
        row_words   = collect_row_words(all_words, anchor, next_anchor)
        cols        = reconstruct_columns(row_words, cm)

        criteria_text   = cols.get("criteria", "") or cols.get("param", "")
        bidder_response = cols.get("response", "")
        docref_col      = cols.get("docref", "")
        marks_col       = cols.get("marks", "")

        max_marks      = _extract_max_marks(marks_col, criteria_text)
        proposed_marks = _extract_proposed_marks(bidder_response, marks_col)
        ev_pages       = extract_evidence_pages(bidder_response + " " + docref_col)

        param_text  = (cols.get("param", "") or criteria_text)
        param_lines = [l.strip() for l in param_text.split() if len(l.strip()) > 2]
        parameter   = " ".join(param_lines[:10])[:100] or f"Criterion {anchor.code}"

        rows.append({
            "item_code": anchor.code, "parent_code": anchor.parent,
            "parameter": parameter, "criteria_text": criteria_text[:2000],
            "max_marks": max_marks, "proposed_marks": proposed_marks,
            "bidder_response": bidder_response[:4000],
            "evidence_pages": ev_pages, "is_sub_item": anchor.is_sub,
            "raw_page": anchor.page,
        })
    return rows


def _parse_markdown_table(markdown: str, page_no: int) -> list:
    """Parse a Docling markdown table into raw row dicts."""
    rows = []
    lines = [l for l in markdown.split("\n") if "|" in l and not re.match(r"^\|[-: ]+\|", l)]
    for i, line in enumerate(lines[1:], 1):  # skip header row
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 3: continue
        # Assume: col0=sno, col1=criteria, col2+=response
        sno_text = cells[0]
        if not re.match(r"^\d{1,2}$", sno_text): continue
        sno = int(sno_text)
        if not (1 <= sno <= 25): continue
        rows.append({
            "item_code": str(sno), "parent_code": "",
            "parameter": cells[1][:100] if len(cells) > 1 else f"Criterion {sno}",
            "criteria_text": cells[1][:2000] if len(cells) > 1 else "",
            "max_marks": None, "proposed_marks": None,
            "bidder_response": " ".join(cells[2:])[:4000] if len(cells) > 2 else "",
            "evidence_pages": [], "is_sub_item": False, "raw_page": page_no,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Scoring integration
# ─────────────────────────────────────────────────────────────────────────────

def score_from_matrix(
    matrix: ComplianceMatrix,
    rfp_criteria: list,
    pre_cached_bands: Optional[dict] = None,
) -> list:
    by_code = {r.item_code: r for r in matrix.rows}

    def _best_row(criterion: dict) -> Optional[ComplianceRow]:
        code = str(criterion.get("item_code", "")).strip()
        if code in by_code: return by_code[code]
        if code and code[:-1] in by_code: return by_code[code[:-1]]
        param = (criterion.get("parameter") or "").lower()
        best_score, best_row = 0.0, None
        for row in matrix.rows:
            words_crit = set(re.findall(r'\b[a-z]{4,}\b', param))
            words_row  = set(re.findall(r'\b[a-z]{4,}\b', row.parameter.lower()))
            if not words_crit or not words_row: continue
            overlap = len(words_crit & words_row) / len(words_crit | words_row)
            if overlap > best_score: best_score, best_row = overlap, row
        return best_row if best_score >= 0.30 else None

    results = []
    for criterion in rfp_criteria:
        max_marks   = int(criterion.get("max_marks") or 0)
        parameter   = criterion.get("parameter", "")
        item_code   = str(criterion.get("item_code", ""))
        formula     = criterion.get("formula_type", "LLM").upper()
        crit_text   = criterion.get("criteria_text", "")
        is_parent   = criterion.get("is_parent", False)
        is_sub_item = criterion.get("is_sub_item", False)

        if max_marks == 0 or is_parent:
            results.append(_zero_result(criterion)); continue

        row = _best_row(criterion)
        if row is None:
            print(f"  [ScoreMatrix] '{parameter[:40]}' — no matching matrix row")
            results.append({**_zero_result(criterion),
                            "justification": "No matching row found in FORM TECH-4 matrix",
                            "gaps": ["Criterion not found in bidder's compliance table"]})
            continue

        value_str = row.extracted_value
        label     = row.extracted_label
        if not value_str:
            value_str, label = extract_bidder_fact(row.bidder_response,
                                                    crit_text or row.criteria_text,
                                                    formula_hint=formula)
        if not value_str:
            print(f"  [ScoreMatrix] '{parameter[:40]}' — value not found")
            results.append({**_zero_result(criterion),
                            "justification": f"Fact not found in bidder response for: {parameter}",
                            "gaps": ["Value not stated in compliance table"],
                            "source_page": row.raw_page})
            continue

        score, steps = _apply_rfp_formula(formula, value_str, crit_text, max_marks,
                                          criterion, pre_cached_bands)

        ev_note = ""
        if row.evidence_pages:
            ev_note = (f"Verified on p.{row.evidence_pages[0]}" if row.verified
                       else f"Cited p.{row.evidence_pages[0]} but not confirmed")
            if not row.verified and score > 0:
                score = round(score * 0.9, 1)
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
            "justification": (f"Score {score}/{max_marks}. Found: {value_str}. "
                              f"{ev_note}. From FORM TECH-4."),
            "strengths":                       [f"{label} | {ev_note}"] if score > 0 else [],
            "gaps":                            ([] if score >= max_marks
                                                else ["Higher value needed for full marks"]),
            "evidence_found":                  score > 0,
            "verified":                        row.verified,
            "evaluation_layer":                "document",
            "requires_live_assessment":        False,
            "requires_comparative_evaluation": False,
            "discrepancies":                   [],
            "source":                          "compliance_matrix",
        })

    return results


def _apply_rfp_formula(formula, value_str, criteria_text, max_marks, criterion, cached_bands):
    num = None
    m = re.search(r"([\d,]+(?:\.\d+)?)", (value_str or "").replace(",", ""))
    if m:
        try: num = float(m.group(1))
        except ValueError: pass

    if num is None:
        if value_str and value_str.lower() not in ("not found", "no", "none"):
            return float(max_marks), f"BINARY: presence → {max_marks}"
        return 0.0, "BINARY: no evidence → 0"

    if formula in ("BAND", "BAND_CR", "BAND_PROJECTS", "BAND_HEADCOUNT", "BAND_YEARS", "STEP"):
        try:
            from core.tq_extractor import _parse_band_table_strict, _apply_band_strict, _apply_step
            bands = _parse_band_table_strict(criteria_text, formula)
            if bands:
                score = _apply_band_strict(bands, num, max_marks, formula)
                if score is not None:
                    return score, f"{formula}: {num} → {score}/{max_marks}"
            if formula == "STEP":
                score = _apply_step(criteria_text, max_marks, num)
                if score is not None:
                    return score, f"STEP: {num} → {score}/{max_marks}"
        except Exception:
            pass

        if cached_bands:
            band_list = (cached_bands.get(criterion.get("parameter", ""), {})
                         .get("bands", []))
            if band_list:
                for band in sorted(band_list, key=lambda b: float(b.get("min") or 0)):
                    lo = float(band.get("min") or 0)
                    hi_raw = band.get("max")
                    hi = float(hi_raw) if hi_raw is not None else float("inf")
                    if lo <= num <= hi:
                        score = round(min(float(band.get("score") or 0), float(max_marks)), 1)
                        return score, f"Cached bands: {num} in [{lo},{hi}] → {score}"

        return round(min(num, float(max_marks)), 1), f"Fallback: {num}"

    if formula == "PER_UNIT":
        rate_m = re.search(r"(\d+(?:\.\d+)?)\s*marks?\s+(?:for|per)\s+(?:each|01|one|per)\s+"
                           r"(?:project|assignment)", criteria_text, re.I)
        if rate_m:
            rate = float(rate_m.group(1))
            return round(min(num * rate, float(max_marks)), 1), f"PER_UNIT: {int(num)}×{rate}"
        return round(min(num, float(max_marks)), 1), "PER_UNIT fallback"

    if formula in ("QUAL", "BINARY"):
        return float(max_marks), f"{formula}: presence → {max_marks}"

    if 0 <= num <= max_marks:
        return num, f"LLM (direct): {num}/{max_marks}"
    return round(min(num, float(max_marks)), 1), f"LLM: capped at {max_marks}"


def _zero_result(criterion: dict) -> dict:
    return {
        "item_code": str(criterion.get("item_code", "")),
        "parameter": criterion.get("parameter", ""),
        "max_marks": int(criterion.get("max_marks") or 0),
        "criteria_text": criterion.get("criteria_text", ""),
        "formula_hint": criterion.get("formula_type", "LLM"),
        "is_sub_item": criterion.get("is_sub_item", False),
        "parent_parameter": criterion.get("parent_parameter", ""),
        "score": 0.0, "score_percentage": 0.0,
        "extracted_value": None, "source_page": None, "evidence_pages": [],
        "scoring_steps": "Zero or parent criterion",
        "justification": "Zero-mark or parent criterion",
        "strengths": [], "gaps": [], "evidence_found": False, "verified": False,
        "evaluation_layer": "document",
        "requires_live_assessment": False,
        "requires_comparative_evaluation": False,
        "discrepancies": [], "source": "compliance_matrix",
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Clean run_tq_evaluation adapter (no Docling on full PDF)
# ─────────────────────────────────────────────────────────────────────────────

def run_tq_evaluation(
    rfp_doc_name: str,
    proposal_path: str,
    proposal_doc_name: str,
    progress_callback=None,
    force_rfp_refresh: bool = False,
) -> dict:
    """
    Full TQ evaluation using the compliance matrix parser.
    Docling is NEVER called on the full PDF — only on detected table pages.
    """
    def _prog(step, pct):
        if progress_callback:
            try: progress_callback(step, pct)
            except Exception: pass
        print(f"[TQ] {pct:3d}% -- {step}")

    _prog("Loading RFP criteria", 10)

    # Find and load RFP criteria
    rfp_path = None
    for search_dir in [Path("./uploads"), Path("./tq_uploads"), Path(".")]:
        candidate = search_dir / rfp_doc_name
        if candidate.exists():
            rfp_path = str(candidate)
            break

    rfp_criteria  = []
    cached_bands  = {}
    grand_total   = 100
    live_marks    = 0
    threshold     = 70.0

    if rfp_path:
        try:
            from core.rfp_cache import load_cache
            cache = load_cache(rfp_path)
            if cache and not force_rfp_refresh:
                rfp_criteria = cache.get("criteria", [])
                cached_bands = cache.get("bands", {})
                grand_total  = cache.get("grand_total", 100)
                live_marks   = cache.get("live_marks", 0)
                threshold    = cache.get("threshold", 70.0)
                print(f"[TQ] Loaded {len(rfp_criteria)} criteria from RFP cache")
        except Exception as e:
            print(f"[TQ] Cache load error (non-fatal): {e}")

        if not rfp_criteria:
            _prog("Extracting RFP marking scheme", 15)
            try:
                from core.tq_extractor_v20 import extract_marking_table
                table = extract_marking_table(rfp_doc_name, force_refresh=force_rfp_refresh)
                rfp_criteria = table.get("criteria", [])
                cached_bands = table.get("bands", {})
                grand_total  = table.get("grand_total_marks", 100)
                live_marks   = table.get("live_assessment_marks", 0)
                threshold    = table.get("qualification_threshold_pct", 70.0)
                print(f"[TQ] Extracted {len(rfp_criteria)} criteria from RFP")
            except Exception as e:
                print(f"[TQ] RFP extraction failed: {e}")

    _prog("Parsing proposal compliance matrix", 30)

    # FIX 3: parse_compliance_matrix handles everything safely
    matrix = parse_compliance_matrix(proposal_path)

    _prog("Scoring criteria", 60)

    # Inject cached bands into criteria
    for c in rfp_criteria:
        param = c.get("parameter", "")
        if param in cached_bands:
            c["_cached_bands"] = cached_bands[param]

    scores = score_from_matrix(matrix, rfp_criteria, pre_cached_bands=cached_bands)

    # Compute totals
    scoreable = sum(s["max_marks"] for s in scores
                    if not s.get("is_sub_item") and s["max_marks"]
                    and not s.get("requires_live_assessment"))
    doc_max_actual = scoreable or (grand_total - live_marks)

    total_score = sum(s.get("score") or 0 for s in scores
                      if s.get("score") is not None
                      and not s.get("requires_live_assessment"))
    total_score = round(total_score, 1)
    pct = round((total_score / doc_max_actual) * 100, 1) if doc_max_actual else 0.0

    # Qualification gate
    qualification = {}
    if threshold:
        min_doc = round(threshold / 100.0 * doc_max_actual, 1)
        passed  = total_score >= min_doc
        qualification = {
            "threshold_pct": float(threshold), "min_doc_marks": min_doc,
            "achieved_doc_marks": total_score, "achieved_pct": pct, "passed": passed,
            "financial_bid_opens": passed,
            "note": (f"Qualified ({total_score}/{doc_max_actual}). Live ({live_marks}) pending."
                     if passed else f"Not qualified — {total_score} < {min_doc} required."),
        }
        print(f"[TQ] Gate: {'QUALIFIED' if passed else 'NOT QUALIFIED'} "
              f"({total_score}/{doc_max_actual}, {pct}%)")

    _prog("Evaluation complete", 100)

    return {
        "evaluation_title":        "Technical Evaluation",
        "grand_total_marks":       grand_total,
        "technical_document_max":  doc_max_actual,
        "scoreable_total":         doc_max_actual,
        "live_assessment_marks":   live_marks,
        "financial_marks":         0,
        "total_scored":            total_score,
        "total_percentage":        pct,
        "final_score_formula":     None,
        "qualification_threshold": threshold,
        "qualification":           qualification,
        "schema_valid":            len(matrix.rows) > 0,
        "schema_warning":          "; ".join(matrix.parse_warnings) or None,
        "global_discrepancies":    [],
        "criteria_structure":      rfp_criteria,
        "scores":                  scores,
        "error":                   (None if matrix.rows
                                    else "No compliance matrix rows found in proposal"),
        "parse_method":            matrix.parse_method,
        "table_pages":             matrix.table_pages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_proposal(proposal_path: str, rfp_criteria: Optional[list] = None) -> None:
    print(f"\n{'='*70}\nCOMPLIANCE MATRIX DIAGNOSTIC\nProposal: {Path(proposal_path).name}\n{'='*70}\n")
    matrix = parse_compliance_matrix(proposal_path)
    print(f"Parse method: {matrix.parse_method}\nTable pages: {matrix.table_pages}")
    print(f"Rows found:  {len(matrix.rows)}")
    if matrix.parse_warnings:
        print(f"Warnings:    {matrix.parse_warnings}")
    print(f"\n{'─'*70}")
    for row in matrix.rows:
        sub = "  " if row.is_sub_item else ""
        ver = "✓" if row.verified else ("⚠" if row.extracted_value else "✗")
        print(f"{sub}{row.item_code:6} {row.parameter[:38]:40} "
              f"{str(row.max_marks):4} {str(row.extracted_value or '—')[:28]:30} {ver}")
    if rfp_criteria:
        results = score_from_matrix(matrix, rfp_criteria)
        total   = sum(r["score"] for r in results if r["score"])
        doc_max = sum(r["max_marks"] for r in results if not rfp_criteria[results.index(r)].get("is_parent"))
        print(f"\n{'─'*70}\nSCORING RESULTS")
        for r in results:
            sc_str = f"{r['score']}/{r['max_marks']}"
            print(f"  [{r['item_code']:4}] {r['parameter'][:45]:45} {sc_str:10}")
        print(f"\n  Total: {total}/{doc_max} ({round(total/doc_max*100,1) if doc_max else 0}%)")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diagnose_proposal(sys.argv[1])
    else:
        print("Usage: python tq_compliance_parser.py <proposal.pdf>")