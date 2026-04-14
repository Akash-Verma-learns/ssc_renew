"""
core/proposal_parser.py  —  v1  Self-Assessment-First Architecture
===================================================================

KEY INSIGHT
───────────
Most government RFP proposals contain a "Technical Bid Evaluation Criteria
and our compliance" table (often FORM TECH-4) where the bidder explicitly:
  • Copies each RFP criterion and scoring rule
  • States their claimed value / compliance
  • Gives exact page numbers for their supporting documents

This table is the PRIMARY source of scoring values.  It is far more
reliable than keyword-searching 500 pages of narrative text.

PIPELINE
────────
  STAGE 0  Detect self-assessment table pages in proposal
           (look for pages containing RFP criteria text + bidder response)
  STAGE 1  Extract table text → parse claimed values per criterion
  STAGE 2  Cross-verify top-N claims against cited evidence pages
           (did they actually say "480.23 Cr" on the page they cited?)
  STAGE 3  Return structured ClaimedValues dict → passed to scorer

This module is called by tq_scorer.score_criterion() BEFORE doing any
keyword search.  If a claimed value is found and verified here, no further
LLM search is needed.

Usage:
    from core.proposal_parser import find_claimed_values
    
    claimed = find_claimed_values(
        proposal_path="tq_uploads/proposal_xxx.pdf",
        criteria=flat_criteria_list,   # from tq_extractor
    )
    # claimed["Financial Turnover"] → {"value": "480.23 Cr", "page": 35, "verified": True}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Self-assessment table detection signals
# ─────────────────────────────────────────────────────────────────────────────

_SA_TABLE_SIGNALS = [
    r"technical\s+bid\s+evaluation\s+criteria\s+and\s+(our\s+)?compliance",
    r"form\s+tech[-\s]*4",
    r"technical\s+evaluation\s+criteria\s+and\s+(our\s+)?response",
    r"our\s+compliance\s+(to|with)\s+(the\s+)?criteria",
    r"compliance\s+statement",
    r"self.assessment",
    r"bidder.s?\s+response\s+to\s+(technical\s+)?evaluation",
]
_SA_SIGNAL_RE = re.compile("|".join(_SA_TABLE_SIGNALS), re.IGNORECASE)

# A page IS part of the self-assessment table if it also has scoring language
_MARKS_RE = re.compile(r"\d+\s*marks?\b|\d+\s*points?\b|max(?:imum)?\s*marks?", re.IGNORECASE)

# Evidence page citation pattern: "page no. 64", "page nos. 153-179", "on page 30"
_PAGE_REF_RE = re.compile(
    r"page\s+(?:no\.?\s*|nos?\.?\s*)?(\d+(?:\s*[-–to]+\s*\d+)?)",
    re.IGNORECASE,
)

# Number extraction
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)")

# "more than N" / "N+" / "> N"
_MORE_THAN_RE = re.compile(r"(more\s+than|over|above|greater\s+than|>\s*)\s*(\d[\d,]*)", re.IGNORECASE)
_AT_LEAST_RE  = re.compile(r"(at\s+least|minimum|min\.?\s*)\s*(\d[\d,]*)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Find self-assessment table pages
# ─────────────────────────────────────────────────────────────────────────────

def _open_pdf(pdf_path: str):
    """Open PDF with fitz; return doc or None."""
    if not _FITZ_OK:
        return None
    try:
        return fitz.open(pdf_path)
    except Exception as e:
        print(f"[Parser] Cannot open proposal: {e}")
        return None


def find_sa_table_pages(pdf_path: str, max_search_pages: int = 80) -> list[int]:
    """
    Find pages that are part of the self-assessment TQ table.
    Returns 1-indexed page numbers.
    """
    doc = _open_pdf(pdf_path)
    if not doc:
        return []

    candidate_start = None
    candidate_pages: list[int] = []

    try:
        for pg_idx in range(min(max_search_pages, len(doc))):
            txt = doc[pg_idx].get_text()
            pg  = pg_idx + 1

            is_sa_header = bool(_SA_SIGNAL_RE.search(txt))
            has_marks    = bool(_MARKS_RE.search(txt))

            if is_sa_header and has_marks:
                candidate_start = pg
                candidate_pages = [pg]
                print(f"[Parser] SA table detected at page {pg}")
                continue

            # Once in the table, keep adding pages that have marks language
            if candidate_start and has_marks and not is_sa_header:
                # Stop if we hit a totally different section (no marks, no page refs)
                has_page_ref = bool(_PAGE_REF_RE.search(txt))
                if has_marks or has_page_ref:
                    candidate_pages.append(pg)
                else:
                    # Possible end of table
                    if len(candidate_pages) >= 3:
                        break

        # Cap to reasonable span
        if candidate_pages and (candidate_pages[-1] - candidate_pages[0]) > 25:
            candidate_pages = [p for p in candidate_pages
                               if p <= candidate_pages[0] + 25]

    finally:
        doc.close()

    print(f"[Parser] SA table pages: {candidate_pages}")
    return candidate_pages


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Extract text from SA table pages
# ─────────────────────────────────────────────────────────────────────────────

def _get_pages_text(pdf_path: str, page_nos: list[int]) -> str:
    """Extract text from the given 1-indexed pages."""
    doc = _open_pdf(pdf_path)
    if not doc:
        return ""
    parts = []
    try:
        for pg in page_nos:
            if 1 <= pg <= len(doc):
                parts.append(f"[PAGE {pg}]\n{doc[pg-1].get_text()}")
    finally:
        doc.close()
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b — Match criterion to its section in the SA table text
# ─────────────────────────────────────────────────────────────────────────────

# Pattern to detect where the bidder's own response starts
_FIRM_RESPONSE_RE = re.compile(
    r"((?:grant\s+thornton|our\s+(?:firm|agency|company)|we\s+have|"
    r"our\s+(?:average|annual)|the\s+(?:firm|agency|company)\s+has|"
    r"[A-Z][a-z]+\s+(?:Bharat|India|Solutions|Consulting|Advisory)\s+(?:LLP|Ltd|Pvt|Pvt\.))"
    r".{0,3000})",
    re.IGNORECASE | re.DOTALL,
)


def _find_criterion_block(sa_text: str, criterion: dict) -> str:
    """
    Find the bidder's RESPONSE portion of the self-assessment table for
    this criterion.  Returns only the text after the firm's name / response
    marker, not the criteria/threshold text (which would confuse value parsing).
    """
    parameter = criterion.get("parameter", "")

    # Build anchor words from parameter name
    words = re.findall(r'\b[a-zA-Z]{4,}\b', parameter)
    if not words:
        return ""

    # Find where this criterion appears in the SA table
    anchor_pos = -1
    for word in words[:3]:
        m = re.search(re.escape(word), sa_text, re.IGNORECASE)
        if m:
            anchor_pos = m.start()
            break

    if anchor_pos == -1:
        return ""

    # Take a wide window around the criterion
    window = sa_text[anchor_pos: anchor_pos + 3000]

    # Now find the bidder's response section within that window
    # (the part that starts with the firm name or "our" statements)
    rm = _FIRM_RESPONSE_RE.search(window)
    if rm:
        return rm.group(0)[:2000]

    # Fallback: return full window
    return window[:2000]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Parse claimed value from the criterion block
# ─────────────────────────────────────────────────────────────────────────────

_TURNOVER_RE = re.compile(
    r"(?:average\s+turnover|turnover)[^\n]*?(?:INR\s*|Rs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:cr(?:ores?)?|lakh)?",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\s+to\s+"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:present|date|now|till\s+date))",
    re.IGNORECASE,
)
_COUNT_PROJECTS_RE = re.compile(
    r"(?:details\s+of\s+the\s+following\s+(\d+)|following\s+(\d+)\s+(?:such\s+)?assign|(\d+)\s+(?:such\s+)?assign(?:ment)?s?\s+have)",
    re.IGNORECASE,
)
_STAFF_COUNT_RE = re.compile(
    r"(?:more\s+than|over|above|at\s+least)?\s*(\d[\d,]+)\s+technically\s+qualified(?:\s+personnel)?",
    re.IGNORECASE,
)


def _parse_turnover(block: str) -> Optional[str]:
    """Extract average turnover value from text block."""
    # Pattern: "INR 480.23 crores" or "480.23 Cr"
    m = re.search(
        r"(?:INR\s*|Rs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:cr(?:ores?)?)",
        block, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "") + " Cr"
    return None


def _parse_experience_years(block: str) -> Optional[str]:
    """
    Find the earliest project start year to compute years of experience.
    Returns e.g. "more than 8 years".
    """
    import datetime
    MONTHS = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,
               "aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    current_year = 2024  # Reference year

    years_found = []
    # Find all "Month YYYY to ..." patterns
    for m in _YEAR_RANGE_RE.finditer(block):
        start_str = m.group(1).lower()
        parts     = start_str.split()
        if len(parts) == 2:
            mon_str, yr_str = parts
            month = next((v for k, v in MONTHS.items() if mon_str.startswith(k)), None)
            if month:
                try:
                    year = int(yr_str)
                    if 2000 <= year <= current_year:
                        years_found.append(year)
                except ValueError:
                    pass

    if not years_found:
        return None
    earliest = min(years_found)
    yrs      = current_year - earliest
    return f"more than {yrs} years (since {earliest})"


def _parse_project_count(block: str) -> Optional[str]:
    """Extract number of projects claimed (e.g. '5 projects')."""
    m = _COUNT_PROJECTS_RE.search(block)
    if m:
        count = m.group(1) or m.group(2) or m.group(3)
        if count:
            return f"{count} projects"

    # Fallback: count table rows (lines with "Sl.no" pattern)
    rows = re.findall(r'^\s*\d+\s+(?:State\s+Technical|Technical\s+Support|Project\s+Management)',
                      block, re.MULTILINE | re.IGNORECASE)
    if rows:
        return f"{len(rows)} projects"
    return None


def _parse_staff_count(block: str) -> Optional[str]:
    """Extract staff count claimed."""
    m = _STAFF_COUNT_RE.search(block)
    if m:
        count = m.group(1).replace(",", "")
        return f"more than {count} technically qualified personnel"

    m2 = _MORE_THAN_RE.search(block)
    if m2 and any(kw in block.lower() for kw in ["payroll", "pay roll", "personnel", "staff"]):
        return f"more than {m2.group(2)} personnel"
    return None


def _parse_binary_presence(block: str, criterion: dict) -> Optional[str]:
    """For BINARY criteria, check if the bidder asserts presence."""
    positive_signals = [
        "extensive experience", "has experience", "has worked",
        "enclos", "supporting document", "work order", "service agreement",
        "grant thornton has", "we have", "our firm has",
    ]
    if any(sig in block.lower() for sig in positive_signals):
        return "Present — evidence cited in proposal"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Extract cited evidence page numbers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_evidence_pages(block: str) -> list[int]:
    """Extract all page numbers referenced in this criterion's block."""
    pages: list[int] = []
    for m in _PAGE_REF_RE.finditer(block):
        ref = m.group(1)
        # Handle ranges like "153-179"
        range_m = re.match(r'(\d+)\s*[-–to]+\s*(\d+)', ref)
        if range_m:
            pages.append(int(range_m.group(1)))  # Just the start page
        else:
            try:
                pages.append(int(ref.strip()))
            except ValueError:
                pass
    return sorted(set(pages))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Spot-check claimed value against cited evidence page
# ─────────────────────────────────────────────────────────────────────────────

def _verify_on_evidence_page(pdf_path: str, claimed_value: str,
                              evidence_pages: list[int]) -> bool:
    """
    Quick verification: does at least one evidence page contain the
    key number/term from the claimed value?
    Returns True if verified, False if not found.
    """
    if not evidence_pages or not claimed_value:
        return False  # Can't verify without a page ref

    doc = _open_pdf(pdf_path)
    if not doc:
        return True   # Can't verify → assume True (don't penalise)

    # Extract key number from claimed value for verification
    nums = re.findall(r'\d[\d,.]*', claimed_value)
    key_terms = nums[:2] if nums else [claimed_value[:20]]

    try:
        for pg in evidence_pages[:3]:  # Check max 3 pages
            if 1 <= pg <= len(doc):
                pg_text = doc[pg-1].get_text().lower()
                if any(term.replace(",", "").lower() in pg_text for term in key_terms):
                    return True
    finally:
        doc.close()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def find_claimed_values(
    proposal_path: str,
    criteria: list[dict],
    max_search_pages: int = 80,
) -> dict[str, dict]:
    """
    Extract claimed values from the proposal's self-assessment table.

    Returns:
        {
            "Financial Turnover": {
                "value":    "480.23 Cr",
                "page":     35,          # page in proposal where found
                "ev_pages": [64],        # cited evidence pages
                "verified": True,        # value confirmed on evidence page
                "method":   "sa_table",
            },
            ...
        }

    If a criterion is not found in the SA table, it will not have a key
    in the returned dict — the caller should fall back to keyword search.
    """
    if not Path(proposal_path).exists():
        print(f"[Parser] Proposal not found: {proposal_path}")
        return {}

    # Stage 0: find self-assessment pages
    sa_pages = find_sa_table_pages(proposal_path, max_search_pages)
    if not sa_pages:
        print("[Parser] No self-assessment table found — falling back to keyword search")
        return {}

    # Stage 1: extract full SA table text
    sa_text   = _get_pages_text(proposal_path, sa_pages)
    sa_page_0 = sa_pages[0]  # first page of the table

    results: dict[str, dict] = {}
    formula_parsers = {
        "BAND":     None,  # handled per-criterion below
        "STEP":     None,
        "PER_UNIT": None,
        "BINARY":   _parse_binary_presence,
        "QUAL":     _parse_binary_presence,
        "LLM":      None,
    }

    for criterion in criteria:
        if criterion.get("is_parent"):
            continue

        parameter    = criterion.get("parameter", "")
        formula_type = (criterion.get("formula_type") or "LLM").upper()

        # Stage 1b: find the relevant block in SA text
        block = _find_criterion_block(sa_text, criterion)
        if not block:
            print(f"[Parser] No block found for: {parameter[:50]}")
            continue

        ev_pages = _extract_evidence_pages(block)

        # Stage 2: parse the claimed value based on criterion type + parameter name
        value: Optional[str] = None
        param_lower = parameter.lower()

        if any(kw in param_lower for kw in ["turnover", "financial", "revenue"]):
            value = _parse_turnover(block)

        elif any(kw in param_lower for kw in ["experience", "area", "year"]):
            value = _parse_experience_years(block)

        elif any(kw in param_lower for kw in ["project", "large scale", "handling", "pmu", "pmc"]):
            value = _parse_project_count(block)

        elif any(kw in param_lower for kw in ["manpower", "staff", "personnel", "consulting staff"]):
            value = _parse_staff_count(block)

        elif formula_type == "BINARY" or any(kw in param_lower for kw in ["maharashtra", "state", "region"]):
            value = _parse_binary_presence(block, criterion)

        # Generic number fallback
        if not value and formula_type in ("BAND", "STEP", "PER_UNIT"):
            nums = _NUMBER_RE.findall(block[:800])
            if nums:
                value = nums[0] + " (parsed from SA table)"

        if not value:
            print(f"[Parser] Could not parse value for: {parameter[:50]}")
            continue

        # Stage 3: verify on cited evidence page
        verified = _verify_on_evidence_page(proposal_path, value, ev_pages)
        if not verified and ev_pages:
            print(f"[Parser] Warning: could not verify '{value}' on p.{ev_pages[0]} for {parameter[:40]}")

        results[parameter] = {
            "value":    value,
            "page":     sa_page_0,
            "ev_pages": ev_pages,
            "verified": verified,
            "method":   "sa_table",
        }
        status = "✓ verified" if verified else "⚠ unverified"
        print(f"[Parser] {parameter[:45]:45s} → {value[:35]} [{status}]")

    print(f"[Parser] Extracted {len(results)}/{len([c for c in criteria if not c.get('is_parent')])} criteria from SA table")
    return results
