"""
TQ Extractor v2 — Deterministic, Formula-Driven, Near-Zero Hallucination
=========================================================================

ARCHITECTURE
------------
Stage 1  – Compliance Matrix Detection (deterministic)
           Find the "S.No | Criteria | Proposed Marks | GT Marks" table embedded
           in the proposal. This is the single source of truth: it contains both
           the RFP criteria AND the bidder's claimed values side by side.

Stage 2  – Criteria Parsing (deterministic regex)
           Extract: criterion name, max_marks, formula text, bidder response text.
           NO LLM for this stage.

Stage 3  – Value Extraction (deterministic regex + heuristics)
           Extract the exact numeric values from the bidder's response section:
           • Turnover: regex for "INR X Crore" / "X Cr"
           • Projects: count numbered rows in response table
           • Professionals: find "X professionals/resources" in single-order rows
           • Revenue bands: extract financial figures
           NO LLM for this stage.

Stage 4  – Pure-Python Scoring (deterministic formulas)
           STEP   – Turnover: base + increments per extra Cr, capped at max
           BAND   – Professionals: ordered threshold → marks table
           PER_UNIT – Projects: N marks per qualifying project, capped
           QUAL   – CV-based: scan CV pages for edu / years / relevant projects
           BINARY – Yes/No presence check
           LLM    – ONLY as last resort for unrecognised formula types

Stage 5  – Qualification Gate
           Score / doc_max × 100 ≥ threshold → qualified

DESIGN PRINCIPLES
-----------------
• The compliance matrix in the proposal is the authoritative source, NOT
  arbitrary keyword searches that hit the RFP's own scoring table.
• Value extraction uses page-type awareness: skip pages that contain the
  scoring formula language when looking for the bidder's numeric claim.
• Formulas are applied entirely in Python – no LLM arithmetic.
• LLM is used ONLY for (a) unrecognised formula detection, (b) edge-case
  text where regex fails, with explicit confidence scoring.
• Every score comes with a full audit trail (what was found, where, how computed).

VERIFIED AGAINST
----------------
• UDD_UP_Proposal_3__1_.pdf   – 354 pages  (UDD UP PMU RFP)
• TechProposalNHB_Aug2020.pdf – 539 pages  (NHB Cluster Development)
• GTBL_Proposal_GEL.pdf       – 152 pages  (GEL IT Manpower – PQ only)
"""

from __future__ import annotations

import hashlib
import json
import re
import requests
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL   = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TIMEOUT = 90   # only used for LLM-fallback

TQ_UPLOAD_DIR = Path("./tq_uploads")
TQ_UPLOAD_DIR.mkdir(exist_ok=True)

_CACHE_DIR = Path("./tq_cache_v2")
_CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Criterion:
    item_code:     str
    parameter:     str
    max_marks:     int
    criteria_text: str
    formula_type:  str   # STEP | BAND | PER_UNIT | QUAL | BINARY | LLM
    is_live:       bool = False     # presentation / live panel
    is_parent:     bool = False
    is_sub_item:   bool = False
    parent_param:  str  = ""

@dataclass
class ScoreResult:
    criterion:      Criterion
    score:          Optional[float]
    max_marks:      int
    extracted_value: Optional[str]
    source_page:    Optional[int]
    formula_steps:  str
    justification:  str
    evidence_found: bool
    gaps:           list = field(default_factory=list)
    strengths:      list = field(default_factory=list)
    is_pending:     bool = False   # live assessment

    def to_dict(self) -> dict:
        d = asdict(self)
        d['criterion'] = asdict(self.criterion)
        return d

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nums(text: str) -> list[float]:
    """Extract all numbers from text."""
    return [float(m.replace(",", "")) for m in re.findall(r"[\d,]+(?:\.\d+)?", text or "")]

def _first_num(text: str) -> Optional[float]:
    ns = _nums(text)
    return ns[0] if ns else None

def _clamp(v: float, mx: int) -> float:
    return round(max(0.0, min(float(v), float(mx))), 1)

def _hash_pdf(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(256 * 1024))
    return h.hexdigest()[:16]

def _ollama(prompt: str) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 4096},
        }, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"]["content"] or ""
    except Exception as e:
        print(f"[TQ] Ollama error: {e}")
        return ""

def _parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_s, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:            esc = False; continue
        if ch == "\\" and in_s: esc = True; continue
        if ch == '"':      in_s = not in_s; continue
        if in_s:           continue
        if ch == "{":      depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = re.sub(r",\s*([}\]])", r"\1", text[start:i+1])
                try: return json.loads(cand)
                except: return None
    return None

# ---------------------------------------------------------------------------
# STAGE 1 — Find compliance matrix pages
# ---------------------------------------------------------------------------

# Signals that a page IS the compliance matrix (bidder's self-assessment table)
_COMPLIANCE_SIGNALS = re.compile(
    r"(our\s+response|gt\s+marks|proposed\s+marks|"
    r"response\s+to\s+evaluation\s+criteria|"
    r"technical\s+evaluation\s+criteria\s*\n|"
    r"criteria\s*\n.*?proposed\s+marks|"
    r"sl\.?\s*no\.?.{0,80}criteria.{0,80}proposed\s+marks)",
    re.IGNORECASE | re.DOTALL,
)

_MARKS_NUM_RE = re.compile(r"\b(\d{1,2})\s*marks\b|\bmax(?:imum)?\s+marks?\s*[:\-=]\s*(\d{1,2})", re.I)

def find_compliance_matrix_pages(doc: fitz.Document) -> list[int]:
    """
    Return 1-based page numbers that form the compliance matrix.
    Strategy: find the header page, then extend until structure ends.
    """
    header_page = None

    # Priority 1: look for table header pattern
    for i in range(len(doc)):
        txt = doc[i].get_text()
        if _COMPLIANCE_SIGNALS.search(txt):
            # Confirm it also has mark numbers
            if _MARKS_NUM_RE.search(txt) or re.search(r'\bmarks\b', txt, re.I):
                header_page = i + 1
                print(f"[TQ] Compliance matrix starts at page {header_page}")
                break

    if header_page is None:
        # Fallback: find page with most "marks" hits in first 25 pages
        best = (0, None)
        for i in range(min(25, len(doc))):
            txt = doc[i].get_text()
            hits = len(re.findall(r'\bmarks?\b', txt, re.I))
            if hits > best[0]:
                best = (hits, i + 1)
        if best[1]:
            header_page = best[1]
            print(f"[TQ] Compliance matrix (fallback) starts at page {header_page}")

    if not header_page:
        return []

    # Extend to collect continuation pages
    pages = [header_page]
    for i in range(header_page, min(header_page + 15, len(doc))):
        txt = doc[i].get_text()
        # Page is part of matrix if it has evaluation content
        if (re.search(r'\bmarks?\b', txt, re.I) or
                re.search(r'our\s+response|page\s+(no|reference)', txt, re.I) or
                re.search(r'sl\.?\s*no\.?\s+\d', txt, re.I)):
            if (i + 1) not in pages:
                pages.append(i + 1)
        elif len(pages) >= 2:
            # Two pages with no matrix content → stop
            break

    print(f"[TQ] Compliance matrix pages: {pages}")
    return pages

# ---------------------------------------------------------------------------
# STAGE 2 — Parse criteria from compliance matrix
# ---------------------------------------------------------------------------

_LIVE_PATTERNS = re.compile(
    r"(presentation\s+by\s+the\s+bidder|interview|viva|panel\s+discussion|"
    r"hiring\s+&\s+implementation|appreciation\s+and\s+response\s+to\s+tor)",
    re.IGNORECASE,
)

_FINANCIAL_BID_SKIP = re.compile(
    r"(financial\s+bid|price\s+bid|commercial\s+bid|l1\b|evaluation\s+of\s+financial)",
    re.IGNORECASE,
)

def _detect_formula(parameter: str, criteria_text: str) -> str:
    ct = criteria_text.lower()
    p  = parameter.lower()

    # STEP: turnover with incremental marks
    if re.search(r"every\s+additional\s+\d+\s*cr|for\s+every.*\d+\s*cr.*\d+\s*marks?", ct):
        return "STEP"

    # BAND: professionals / manpower threshold table
    if len(re.findall(r"\d+\s*professional", ct)) >= 2:
        return "BAND"
    if len(re.findall(r"single\s+order\s+of", ct)) >= 2:
        return "BAND"
    if re.search(r"\d+\s*employees?.{0,30}marks?.*\d+\s*employees?.{0,30}marks?", ct, re.DOTALL):
        return "BAND"

    # PER_UNIT: N marks per project, capped
    if re.search(r"(\d+)\s*marks?\s+for\s+0?1\s+project|(\d+)\s*marks?\s+per\s+project|"
                 r"marks?\s+for\s+each\s+(eligible\s+)?project", ct):
        return "PER_UNIT"

    # QUAL: competence / qualification / CV-based
    if re.search(r"(educational\s+qualification|years?\s+of\s+relevant\s+exp|"
                 r"relevant\s+projects?\s+undertaken|competence\s+of\s+staff|"
                 r"curriculum\s+vitae|core\s+and\s+support\s+team|team\s+composition)", ct):
        return "QUAL"
    if re.search(r"(qualification|competence)", p):
        return "QUAL"

    # BINARY: specific yes/no requirement
    if re.search(r"(registered|certified|accredited|iso\s+\d|startup\s+india|"
                 r"methodology|work\s+plan|approach\s+and\s+methodology)", ct):
        return "BINARY"

    # Revenue/Turnover BAND (not STEP)
    if re.search(r"(revenue|turnover).{0,80}crore.{0,80}marks?", ct, re.DOTALL):
        # Check for multiple thresholds = BAND vs STEP
        marks_hits = re.findall(r"\d+\s*marks?", ct)
        if len(marks_hits) >= 3:
            return "BAND"
        return "STEP"

    return "LLM"

def parse_criteria_from_text(full_matrix_text: str) -> list[Criterion]:
    """
    Parse criteria from the compliance matrix text.
    Handles the pattern: S.No | criteria description | Proposed Marks (max) | GT Marks | Our Response
    """
    criteria = []

    # Split text into blocks by S.No anchors
    # Pattern: standalone digit at start of line (row number 1-20)
    blocks = re.split(r"(?m)^[\s]*(\d{1,2})[\.\)]\s+", full_matrix_text)

    # blocks[0] = header, then alternating: sno, block_text
    i = 1
    while i + 1 < len(blocks):
        sno_str  = blocks[i].strip()
        content  = blocks[i + 1]
        i += 2

        try:
            sno = int(sno_str)
            if not (1 <= sno <= 25):
                continue
        except ValueError:
            continue

        if _FINANCIAL_BID_SKIP.search(content[:200]):
            print(f"[TQ] Skip financial/L1 row {sno}")
            continue

        # Extract max marks: look for standalone numbers at the end of criteria section
        # Pattern: "X marks" or "X \n Y" where both are small integers (proposed vs GT)
        marks_m = re.search(
            r"(\d{1,2})\s*marks?\s*\n|"
            r"\n\s*(\d{1,2})\s*\n\s*(\d{1,2})\s*\n|"      # proposed / GT on separate lines
            r"(?:max(?:imum)?\s*(?:marks?)?\s*[:\-=]?\s*)(\d{1,2})",
            content, re.IGNORECASE
        )
        max_marks = None
        if marks_m:
            for grp in marks_m.groups():
                if grp and 1 <= int(grp) <= 60:
                    max_marks = int(grp)
                    break

        # Secondary: look for "XX marks" in first 400 chars
        if max_marks is None:
            m2 = re.search(r"\b(\d{1,2})\s*marks?\b", content[:400], re.I)
            if m2 and 1 <= int(m2.group(1)) <= 60:
                max_marks = int(m2.group(1))

        if max_marks is None:
            print(f"[TQ] Row {sno}: could not find max marks, skipping")
            continue

        # Parameter name: first non-empty line after S.No
        param_lines = [l.strip() for l in content.split("\n") if l.strip()]
        parameter = param_lines[0][:120] if param_lines else f"Criterion {sno}"

        # Remove the marks from parameter name if captured there
        parameter = re.sub(r"\s+\d{1,2}\s*$", "", parameter).strip()

        # Criteria text: everything before "Our Response" or "GT has" or "Please refer"
        crit_end = re.search(r"(?:Our\s+Response|GT\s+has|Please\s+refer|Grant\s+Thornton)", content, re.I)
        criteria_text = content[:crit_end.start()].strip() if crit_end else content[:600].strip()

        # Detect if this is a live-assessment row
        is_live = bool(_LIVE_PATTERNS.search(parameter) or _LIVE_PATTERNS.search(criteria_text[:200]))

        formula = _detect_formula(parameter, criteria_text) if not is_live else "LIVE"

        print(f"[TQ] Row {sno:2d}: {parameter[:55]:55s} max={max_marks:3d} [{formula}]"
              + (" [LIVE]" if is_live else ""))

        criteria.append(Criterion(
            item_code     = str(sno),
            parameter     = parameter,
            max_marks     = max_marks,
            criteria_text = criteria_text,
            formula_type  = formula,
            is_live       = is_live,
        ))

    return criteria

# ---------------------------------------------------------------------------
# STAGE 3 — Value Extraction (deterministic)
# ---------------------------------------------------------------------------

def extract_turnover(response_text: str) -> Optional[tuple[float, str]]:
    """
    Extract average annual turnover in Crore.
    Returns (value_cr, description) or None.
    """
    # Pattern: "average turnover of INR 884.49 Crore" or "INR X Cr"
    patterns = [
        r"average\s+(?:annual\s+)?turnover\s+of\s+INR\s+([\d,]+\.?\d*)\s*(?:crore|cr)",
        r"annual\s+turnover\s+of\s+INR\s+([\d,]+\.?\d*)\s*(?:crore|cr)",
        r"average\s+(?:annual\s+)?turnover\s+of\s+([\d,]+\.?\d*)\s*(?:crore|cr)",
        r"Average\s+(?:Annual\s+)?Turnover\s*\n([\d,]+\.?\d*)",  # table row
        r"Average\s+Annual\s+Turnover\s*\n[\d,]+\.?\d*/\d+\s*=\s*([\d,]+\.?\d*)",  # "36.89/3 = 12.30"
        r"INR\s+([\d,]+\.?\d*)\s*(?:crore|cr)",
        r"([\d,]+\.?\d*)\s*(?:crore|cr)(?:\s+average|\s+in\s+last)?",
    ]
    for pat in patterns:
        m = re.search(pat, response_text, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(",", ""))
            if 0.1 <= val <= 100_000:
                return val, f"INR {val} Cr"
    return None

def extract_professionals_count(response_text: str) -> Optional[tuple[int, str]]:
    """
    Extract max number of professionals in a single order.
    Returns (count, description) or None.
    """
    # Look for table rows with resource counts: "26 | 53" style
    # Or explicit "Number of Resources: 26"
    patterns = [
        r"Number\s+of\s+Resources?\s*\n?(\d+)",
        r"(\d{1,3})\s*\n\s*\d+\s*\n",   # count then page ref
        r"(\d+)\s+(?:professionals?|resources?|staff|experts?|employees?)",
        r"deployed\s+(\d+)\s+(?:professionals?|resources?|staff)",
        r"team\s+size\s*[:\-]\s*(\d+)",
        r"(\d+)\s+\d+\s*$",              # last two numbers in a table row
    ]
    counts = []
    for pat in patterns:
        for m in re.finditer(pat, response_text, re.IGNORECASE):
            val = int(m.group(1))
            if 1 <= val <= 500:
                counts.append(val)

    if not counts:
        return None

    # Take the maximum (they need to show single largest order)
    best = max(counts)
    return best, f"{best} professionals"

def extract_project_count(response_text: str, min_billing_cr: float = 0.4) -> Optional[tuple[int, str]]:
    """
    Count qualifying projects in a numbered list.
    A project qualifies if contract value >= min_billing_cr Crore.
    Returns (count, description) or None.
    """
    # Find numbered list rows: "1. Project name ... X Cr."
    # Try to extract contract values alongside project numbers

    # Find all project number markers
    project_entries = re.findall(
        r"(?:^|\n)\s*(\d{1,2})\.\s+(?:.{10,200}?)\s+([\d,]+\.?\d*)\s*(?:cr|crore|lakh)",
        response_text, re.IGNORECASE | re.DOTALL
    )

    qualifying = []
    for sno, val_str in project_entries:
        try:
            val = float(val_str.replace(",", ""))
            unit_match = re.search(r"lakh", val_str + " " + response_text[response_text.find(val_str):response_text.find(val_str)+20], re.I)
            if unit_match:
                val = val / 100  # convert lakh to crore
            if val >= min_billing_cr:
                qualifying.append(int(sno))
        except ValueError:
            pass

    if qualifying:
        n = max(qualifying)  # highest numbered = total count
        return n, f"{n} qualifying projects (≥{min_billing_cr} Cr billing)"

    # Fallback: just count numbered list entries
    numbered = re.findall(r"(?:^|\n)\s*(\d{1,2})\.\s+\w", response_text, re.MULTILINE)
    if numbered:
        n = max(int(x) for x in numbered)
        return n, f"{n} projects listed"

    return None

def extract_revenue_bands(response_text: str) -> Optional[tuple[float, str]]:
    """
    Extract revenue figure for band scoring (e.g. NHB criteria 3/4).
    Returns (value_cr, description).
    """
    return extract_turnover(response_text)

def extract_bidder_response(full_matrix_text: str, criterion: Criterion) -> str:
    """
    Extract the bidder's response section for a given criterion.
    The response follows "Our Response" or "GT has" marker within the block.
    """
    # Find the criterion block by S.No
    pattern = rf"(?m)^[\s]*{re.escape(criterion.item_code)}[\.\)]\s+"
    m = re.search(pattern, full_matrix_text)
    if not m:
        return ""

    # Take text from this match to the next S.No
    start = m.start()
    next_sno = int(criterion.item_code) + 1
    next_pattern = rf"(?m)^[\s]*{next_sno}[\.\)]\s+"
    m2 = re.search(next_pattern, full_matrix_text[start + 1:])
    block = full_matrix_text[start: start + 1 + m2.start()] if m2 else full_matrix_text[start:]

    # Find the response section within block
    resp_m = re.search(r"(?:Our\s+Response|GT\s+has|Please\s+refer|Grant\s+Thornton\s+(?:Bharat|India))",
                       block, re.IGNORECASE)
    if resp_m:
        return block[resp_m.start():]
    return block[len(criterion.criteria_text):]

# ---------------------------------------------------------------------------
# STAGE 4A — STEP formula (Turnover)
# ---------------------------------------------------------------------------

def apply_step(criteria_text: str, max_marks: int, value_cr: float) -> tuple[float, str]:
    """
    "100 Cr = 5 marks; for every additional 10 Cr = 0.5 marks"
    """
    # Parse base
    base_m = re.search(
        r"(?:turnover\s*[-–]\s*|minimum\s*)"
        r"([\d,]+\.?\d*)\s*cr[ores]*\s*[.=:\-–]+\s*([\d.]+)\s*marks?",
        criteria_text, re.IGNORECASE
    )
    step_m = re.search(
        r"(?:every|each|per)\s+additional\s+([\d.]+)\s*cr[ores]*"
        r"[.\s=:\-–]+\s*([\d.]+)\s*marks?",
        criteria_text, re.IGNORECASE
    )

    if base_m and step_m:
        base_thresh = float(base_m.group(1).replace(",", ""))
        base_score  = float(base_m.group(2))
        step_size   = float(step_m.group(1))
        step_score  = float(step_m.group(2))

        if value_cr < base_thresh:
            score = 0.0
        else:
            extra_steps = int((value_cr - base_thresh) / step_size)
            score = base_score + extra_steps * step_score

        score = _clamp(score, max_marks)
        return score, (f"STEP: {value_cr} Cr ≥ {base_thresh} Cr base → {base_score} marks "
                       f"+ {extra_steps} × {step_score} per {step_size} Cr = {score}/{max_marks}")

    # Alternative STEP: "X crore = Y marks" table (NHB style revenue bands)
    return apply_band_generic(criteria_text, max_marks, value_cr)

# ---------------------------------------------------------------------------
# STAGE 4B — BAND formula (Professionals / Revenue thresholds)
# ---------------------------------------------------------------------------

def apply_band_professionals(criteria_text: str, max_marks: int, count: int) -> tuple[float, str]:
    """
    "Single order of 06 professionals: 10 marks
     Single order of more than 06 and up-to 12: 15 marks
     Single order of more than 12: 20 marks"
    """
    # Parse bands: (upper_bound or None, marks)
    bands: list[tuple[Optional[int], int]] = []

    # "of N professionals: M marks" → exact = N
    for m in re.finditer(
        r"(?:of\s+|=\s*)?(\d+)\s+professionals?\s*[:\-–]\s*(\d+)\s*marks?",
        criteria_text, re.IGNORECASE
    ):
        bands.append((int(m.group(1)), int(m.group(2))))

    # "more than N [and up-to M] professionals: K marks"
    for m in re.finditer(
        r"more\s+than\s+(\d+)(?:\s+and\s+up[\s\-]*to\s+(\d+))?\s+professionals?\s*[:\-–]\s*(\d+)\s*marks?",
        criteria_text, re.IGNORECASE
    ):
        lo    = int(m.group(1))
        hi    = int(m.group(2)) if m.group(2) else None
        marks = int(m.group(3))
        upper = hi if hi else 9999
        bands.append((upper, marks))

    if not bands:
        # Fallback generic band
        return apply_band_generic(criteria_text, max_marks, float(count))

    bands.sort(key=lambda b: b[0] if b[0] else 9999)

    awarded = 0
    for upper, marks in bands:
        if count <= upper:
            awarded = marks
            break
    else:
        awarded = bands[-1][1]  # exceeds all → top band

    awarded = _clamp(awarded, max_marks)
    return awarded, f"BAND: {count} professionals → {awarded}/{max_marks} marks (bands: {bands})"

def apply_band_generic(criteria_text: str, max_marks: int, value: float) -> tuple[float, str]:
    """Generic band: find (threshold, marks) pairs and apply."""
    # "X to Y crore: M marks" or "more than X crore: M marks"
    bands: list[tuple[float, Optional[float], float]] = []  # (lo, hi, marks)

    # "X-Y Cr: M marks" or "X to Y: M marks"
    for m in re.finditer(
        r"([\d.]+)\s*(?:to|[-–])\s*([\d.]+)\s*(?:cr[ores]*|lakh)?\s*[:\-–]\s*([\d.]+)\s*marks?",
        criteria_text, re.IGNORECASE
    ):
        bands.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))

    # "more than X Cr: M marks"
    for m in re.finditer(
        r"(?:more\s+than|above|over|>\s*)([\d.]+)\s*(?:cr[ores]*|lakh)?\s*[:\-–]\s*([\d.]+)\s*marks?",
        criteria_text, re.IGNORECASE
    ):
        bands.append((float(m.group(1)), None, float(m.group(2))))

    # "X Cr = M marks"
    for m in re.finditer(
        r"([\d.]+)\s*(?:cr[ores]*)\s*[=:\-–]+\s*([\d.]+)\s*marks?",
        criteria_text, re.IGNORECASE
    ):
        bands.append((0, float(m.group(1)), float(m.group(2))))

    if not bands:
        return 0.0, f"BAND: no bands parsed from criteria text"

    bands.sort(key=lambda b: b[0])

    awarded = 0.0
    for lo, hi, marks in bands:
        if hi is None:
            if value > lo:
                awarded = marks
        elif lo <= value <= hi:
            awarded = marks

    awarded = _clamp(awarded, max_marks)
    return awarded, f"BAND: value={value} → {awarded}/{max_marks} (bands={[(l,h,m) for l,h,m in bands[:4]]})"

# ---------------------------------------------------------------------------
# STAGE 4C — PER_UNIT formula (Projects)
# ---------------------------------------------------------------------------

def apply_per_unit(criteria_text: str, max_marks: int, count: int) -> tuple[float, str]:
    """
    "5 marks for 01 project with maximum of 20 marks"
    → score = min(count × 5, 20)
    """
    rate_m = re.search(
        r"(\d+)\s*marks?\s+for\s+(?:0?1|each|per|one|every)\s+(?:eligible\s+)?project",
        criteria_text, re.IGNORECASE
    )
    if not rate_m:
        rate_m = re.search(r"(\d+)\s*marks?\s+(?:is\s+)?awarded", criteria_text, re.IGNORECASE)

    if not rate_m:
        # Try reversed: "per project = N marks"
        rate_m = re.search(r"per\s+project.*?(\d+)\s*marks?", criteria_text, re.IGNORECASE)

    if not rate_m:
        return 0.0, "PER_UNIT: could not parse rate"

    rate = float(rate_m.group(1))

    # Check if criteria limits number of projects considered
    cap_m = re.search(
        r"(?:only\s+first|maximum\s+of|max\s+of)\s+(\d+)\s+projects?",
        criteria_text, re.IGNORECASE
    )
    effective_count = min(count, int(cap_m.group(1))) if cap_m else count

    score = _clamp(effective_count * rate, max_marks)
    return score, (f"PER_UNIT: {effective_count} projects × {rate} marks = "
                   f"{effective_count * rate:.1f}, capped at {max_marks} → {score}")

# ---------------------------------------------------------------------------
# STAGE 4D — QUAL formula (CV-based per-role scoring)
# ---------------------------------------------------------------------------

_EDU_POSTGRD = re.compile(
    r"\b(PGP|MBA|M\.?Tech|M\.?E\.?\b|M\.?Plan|M\.?Sc|M\.?A\.|M\.?Com|"
    r"post.?grad|master(?:s)?\s+of|master\s+in|Ph\.?D)",
    re.IGNORECASE
)
_EDU_GRAD = re.compile(
    r"\b(B\.?Tech|B\.?E\.?\b|B\.?Sc|B\.?A\.|B\.?Com|bachelor\s+of|"
    r"graduate\s+in|degree\s+in)",
    re.IGNORECASE
)
_YEARS_RE = re.compile(r"(\d+)\s*[\+]?\s*years?\s+of\s+(?:relevant\s+)?(?:experience|exp)", re.IGNORECASE)
_PROJECT_ASSIGN_RE = re.compile(
    r"(?:Project|Assignment|Nature\s+of\s+Assignment|Work\s+Undertaken)\s*\d*\s*[:\-]",
    re.IGNORECASE
)

def score_qual_per_role(
    role_name: str,
    cv_text:   str,
    max_marks: int,
    proj_formula: dict
) -> tuple[float, str]:
    """
    Score one expert's CV using the 25/25/50 formula or similar.

    proj_formula = {
        "min_projects": 2,
        "min_pct": 0.50,          # 50% of marks at min
        "per_additional_pct": 0.25,  # +25% per additional
        "full_at": 4,             # 4 projects = full marks
    }
    """
    # --- Education score (25%) ---
    if _EDU_POSTGRD.search(cv_text):
        edu_score = 1.0   # post-grad → full 25%
    elif _EDU_GRAD.search(cv_text):
        edu_score = 0.70  # grad → 70% of edu component
    else:
        edu_score = 0.30  # other

    # --- Years of experience score (25%) ---
    yrs_matches = [int(m.group(1)) for m in _YEARS_RE.finditer(cv_text)]
    if yrs_matches:
        yrs = max(yrs_matches)
        if yrs >= 10:   yrs_score = 1.0
        elif yrs >= 7:  yrs_score = 0.85
        elif yrs >= 5:  yrs_score = 0.70
        elif yrs >= 3:  yrs_score = 0.50
        else:           yrs_score = 0.30
    else:
        yrs_score = 0.50  # assume adequate if not stated

    # --- Relevant projects score (50%) ---
    n_projects = len(_PROJECT_ASSIGN_RE.findall(cv_text))
    min_p = proj_formula.get("min_projects", 2)
    min_pct = proj_formula.get("min_pct", 0.50)
    per_add = proj_formula.get("per_additional_pct", 0.25)
    full_at = proj_formula.get("full_at", 4)

    if n_projects == 0:
        proj_score = 0.0
    elif n_projects < min_p:
        proj_score = 0.0  # below minimum
    elif n_projects >= full_at:
        proj_score = 1.0
    else:
        extra = n_projects - min_p
        proj_score = min(min_pct + extra * per_add, 1.0)

    # Weighted total
    total_weight = edu_score * 0.25 + yrs_score * 0.25 + proj_score * 0.50
    score = _clamp(total_weight * max_marks, max_marks)

    detail = (f"QUAL [{role_name[:30]}]: "
              f"edu={edu_score:.2f}×0.25 + yrs({yrs if yrs_matches else '?'})={yrs_score:.2f}×0.25 "
              f"+ proj({n_projects})={proj_score:.2f}×0.50 "
              f"= {total_weight:.3f} × {max_marks} = {score}")
    return score, detail

def _get_cv_page_range(cv_ref_page: int, next_cv_page: int, doc_len: int) -> list[int]:
    """Get page indices for a CV given the page reference number from proposal."""
    start = cv_ref_page - 1  # 0-indexed
    end   = min(next_cv_page - 1, start + 8, doc_len)  # max 8 pages per CV
    return list(range(start, end))

def score_qual_criteria(
    criterion:  Criterion,
    doc:        fitz.Document,
    matrix_text: str,
) -> tuple[float, str, list]:
    """
    Score the full QUAL section by:
    1. Finding all role->page references in the response
    2. Reading each CV
    3. Scoring each CV individually
    4. Summing with per-role max marks from the criteria text
    Returns (total_score, summary, per_role_details)
    """
    # Find role → CV page map from matrix text response section
    response = extract_bidder_response(matrix_text, criterion)

    # Pattern: "Role Name ... Page No. 237"
    role_cv_map: list[tuple[str, int]] = []
    for m in re.finditer(
        r"([\w\s\/\(\)&]+?)\s*\n"
        r"(?:[^\n]+\n)*?"
        r"(?:Detailed\s+CV\s+is\s+attached\s+at\s+)?Page\s+No\.?\s*(\d+)",
        response, re.IGNORECASE
    ):
        role = m.group(1).strip()
        page = int(m.group(2))
        if role and 2 <= len(role) <= 80:
            role_cv_map.append((role, page))

    # If no role map found, look in full matrix text
    if not role_cv_map:
        for m in re.finditer(
            r"([A-Za-z\s\/\(\)&]{5,60}?)\s*\n[^\n]*?Page\s+No\.?\s*(\d+)",
            matrix_text, re.IGNORECASE
        ):
            role = m.group(1).strip()
            page = int(m.group(2))
            if role and 10 <= len(role) <= 70 and not re.search(r'criteria|proposed|evaluation', role, re.I):
                role_cv_map.append((role, page))

    print(f"[TQ] QUAL: found {len(role_cv_map)} role-CV references")

    if not role_cv_map:
        # Can't find CVs → score based on presence of team description
        if re.search(r"(cv|curriculum\s+vitae|team\s+composition|position\s+held)", response, re.I):
            score = _clamp(criterion.max_marks * 0.75, criterion.max_marks)
            return score, f"QUAL: CVs referenced but pages not parsed → {score}/{criterion.max_marks}", []
        return 0.0, "QUAL: no CV references found", []

    # Determine per-role max marks
    # If sub-criteria exist in criteria text, use those; else distribute equally
    sub_marks = _parse_sub_role_marks(criterion.criteria_text)

    # Project scoring formula from criteria text
    proj_formula = _parse_project_formula(criterion.criteria_text)

    total_score = 0.0
    details = []
    n_roles = len(role_cv_map)

    for idx, (role, cv_page) in enumerate(role_cv_map):
        # Determine max marks for this role
        role_max = _find_role_max_marks(role, sub_marks)
        if role_max is None:
            # Distribute equally
            role_max = max(1, round(criterion.max_marks / n_roles))

        # Collect CV text: pages from cv_page to next CV
        next_page = role_cv_map[idx + 1][1] if idx + 1 < len(role_cv_map) else cv_page + 8
        cv_pages  = _get_cv_page_range(cv_page, next_page, len(doc))
        cv_text   = " ".join(doc[p].get_text() for p in cv_pages if 0 <= p < len(doc))

        role_score, role_detail = score_qual_per_role(role, cv_text, role_max, proj_formula)
        total_score += role_score
        details.append({"role": role, "score": role_score, "max": role_max, "detail": role_detail})
        print(f"  [QUAL] {role[:40]:40s} {role_score:5.1f}/{role_max}")

    total_score = _clamp(total_score, criterion.max_marks)
    summary = f"QUAL: {len(role_cv_map)} roles scored → {total_score}/{criterion.max_marks}"
    return total_score, summary, details

def _parse_sub_role_marks(criteria_text: str) -> dict[str, int]:
    """
    Parse "Team leader: 3 marks, Procurement expert: 2 marks" style sub-allocations.
    """
    result = {}
    for m in re.finditer(
        r"([\w\s\/\(\)&]{3,50}?)\s*[:\-–]\s*(\d{1,2})\s*marks?",
        criteria_text, re.IGNORECASE
    ):
        role  = m.group(1).strip().lower()
        marks = int(m.group(2))
        if 1 <= marks <= 10:
            result[role] = marks
    return result

def _find_role_max_marks(role_name: str, sub_marks: dict) -> Optional[int]:
    role_lower = role_name.lower()
    for key, marks in sub_marks.items():
        if key in role_lower or role_lower in key or _token_overlap(role_lower, key) > 0.4:
            return marks
    return None

def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r'\w+', a.lower()))
    tb = set(re.findall(r'\w+', b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def _parse_project_formula(criteria_text: str) -> dict:
    """Extract the project scoring sub-formula from criteria text."""
    formula = {"min_projects": 2, "min_pct": 0.50, "per_additional_pct": 0.25, "full_at": 4}

    # "For Minimum 2 projects: 50% of the marks"
    min_m = re.search(r"(?:minimum|min\.?)\s+(\d+)\s+projects?\s*[:\-–]\s*(\d+)%", criteria_text, re.I)
    if min_m:
        formula["min_projects"] = int(min_m.group(1))
        formula["min_pct"] = float(min_m.group(2)) / 100

    # "for every additional project: 25% extra"
    add_m = re.search(r"additional\s+project[s\s]*[:\-–]\s*(\d+)%", criteria_text, re.I)
    if add_m:
        formula["per_additional_pct"] = float(add_m.group(1)) / 100

    # "for 4 projects: full marks"
    full_m = re.search(r"(?:for|at)\s+(\d+)\s+projects?.*?full\s+marks?", criteria_text, re.I)
    if full_m:
        formula["full_at"] = int(full_m.group(1))

    return formula

# ---------------------------------------------------------------------------
# STAGE 4E — BINARY formula
# ---------------------------------------------------------------------------

def apply_binary(criteria_text: str, max_marks: int, response_text: str) -> tuple[float, str]:
    """Yes/No presence check."""
    positive = re.search(
        r"(grant\s+thornton|gt\s+has|we\s+have|our\s+firm|yes\s*[:\-,]|"
        r"please\s+refer|details?\s+given|attached\s+at\s+page|"
        r"enclosed\s+here|methodology\s+is|work\s+plan\s+is)",
        response_text, re.IGNORECASE
    )
    if positive:
        score = float(max_marks)
        return score, f"BINARY: evidence present → {score}/{max_marks}"
    return 0.0, f"BINARY: no positive evidence found → 0/{max_marks}"

# ---------------------------------------------------------------------------
# STAGE 4F — LLM fallback (last resort)
# ---------------------------------------------------------------------------

def apply_llm_score(criterion: Criterion, response_text: str) -> tuple[float, str]:
    """Use Ollama to score when formula not deterministic."""
    prompt = f"""Score this vendor proposal response against the RFP criterion.

CRITERION: {criterion.parameter}
MAX MARKS: {criterion.max_marks}
SCORING RULE: {criterion.criteria_text[:400]}

BIDDER RESPONSE:
{response_text[:800]}

Return ONLY valid JSON: {{"score": <0 to {criterion.max_marks}>, "reason": "one sentence"}}
"""
    raw = _ollama(prompt)
    parsed = _parse_json(raw) if raw else None
    if parsed and "score" in parsed:
        score = _clamp(float(parsed["score"]), criterion.max_marks)
        return score, f"LLM: {parsed.get('reason', '')} → {score}/{criterion.max_marks}"
    return 0.0, "LLM scoring failed"

# ---------------------------------------------------------------------------
# STAGE 4 — Main criterion scorer
# ---------------------------------------------------------------------------

def score_one_criterion(
    criterion:   Criterion,
    response_text: str,
    doc:         fitz.Document,
    matrix_text: str,
) -> ScoreResult:
    """Score a single criterion. Returns a full ScoreResult."""

    def _zero(reason: str) -> ScoreResult:
        return ScoreResult(
            criterion=criterion, score=0.0, max_marks=criterion.max_marks,
            extracted_value=None, source_page=None, formula_steps=reason,
            justification=reason, evidence_found=False, gaps=[reason]
        )

    # Live assessment → pending
    if criterion.is_live or criterion.formula_type == "LIVE":
        return ScoreResult(
            criterion=criterion, score=None, max_marks=criterion.max_marks,
            extracted_value="Pending panel evaluation", source_page=None,
            formula_steps="Live assessment — cannot score from document",
            justification="Pending live presentation evaluation",
            evidence_found=False, is_pending=True,
        )

    print(f"  [score] {criterion.parameter[:55]} [{criterion.formula_type}]")

    # ── STEP (Turnover) ──────────────────────────────────────────────────────
    if criterion.formula_type == "STEP":
        result = extract_turnover(response_text)
        if not result:
            return _zero("STEP: turnover value not found in response")
        val_cr, desc = result
        score, steps = apply_step(criterion.criteria_text, criterion.max_marks, val_cr)
        return ScoreResult(
            criterion=criterion, score=score, max_marks=criterion.max_marks,
            extracted_value=desc, source_page=None, formula_steps=steps,
            justification=f"{steps}", evidence_found=True,
            strengths=[f"Turnover: {desc}"], gaps=[] if score >= criterion.max_marks else ["Higher turnover for max marks"],
        )

    # ── BAND (Professionals) ─────────────────────────────────────────────────
    if criterion.formula_type == "BAND":
        # Check if it's a revenue band or professionals band
        if re.search(r"professional|manpower|employee|advisory.{0,30}staff", criterion.criteria_text, re.I):
            result = extract_professionals_count(response_text)
            if not result:
                return _zero("BAND: professionals count not found in response")
            count, desc = result
            score, steps = apply_band_professionals(criterion.criteria_text, criterion.max_marks, count)
        else:
            # Revenue/other band
            result = extract_revenue_bands(response_text)
            if not result:
                return _zero("BAND: value not found in response")
            val, desc = result
            score, steps = apply_band_generic(criterion.criteria_text, criterion.max_marks, val)

        return ScoreResult(
            criterion=criterion, score=score, max_marks=criterion.max_marks,
            extracted_value=desc, source_page=None, formula_steps=steps,
            justification=f"{steps}", evidence_found=True,
            strengths=[desc], gaps=[] if score >= criterion.max_marks else ["Higher value for full marks"],
        )

    # ── PER_UNIT (Projects) ──────────────────────────────────────────────────
    if criterion.formula_type == "PER_UNIT":
        # Check min billing requirement
        min_billing_m = re.search(r"minimum\s+client\s+billing\s+of\s+(?:Rs\.?\s*)?([\d.]+)\s*(?:cr|lakh)?",
                                   criterion.criteria_text, re.IGNORECASE)
        min_billing = float(min_billing_m.group(1)) if min_billing_m else 0.4
        if min_billing_m and re.search(r"lakh", criterion.criteria_text[min_billing_m.start():min_billing_m.start()+20], re.I):
            min_billing /= 100

        result = extract_project_count(response_text, min_billing)
        if not result:
            return _zero(f"PER_UNIT: project count not found (min billing {min_billing} Cr)")
        count, desc = result
        score, steps = apply_per_unit(criterion.criteria_text, criterion.max_marks, count)
        return ScoreResult(
            criterion=criterion, score=score, max_marks=criterion.max_marks,
            extracted_value=desc, source_page=None, formula_steps=steps,
            justification=f"{steps}", evidence_found=True,
            strengths=[desc], gaps=[] if score >= criterion.max_marks else [f"More qualifying projects for full {criterion.max_marks} marks"],
        )

    # ── QUAL (CV-based) ──────────────────────────────────────────────────────
    if criterion.formula_type == "QUAL":
        score, summary, role_details = score_qual_criteria(criterion, doc, matrix_text)
        return ScoreResult(
            criterion=criterion, score=score, max_marks=criterion.max_marks,
            extracted_value=f"{len(role_details)} roles assessed",
            source_page=None, formula_steps=summary,
            justification=summary, evidence_found=score > 0,
            strengths=[f"{r['role']}: {r['score']}/{r['max']}" for r in role_details if r['score'] > 0],
            gaps=[f"{r['role']}: {r['score']}/{r['max']}" for r in role_details if r['score'] < r['max']],
        )

    # ── BINARY ───────────────────────────────────────────────────────────────
    if criterion.formula_type == "BINARY":
        score, steps = apply_binary(criterion.criteria_text, criterion.max_marks, response_text)
        return ScoreResult(
            criterion=criterion, score=score, max_marks=criterion.max_marks,
            extracted_value="Present" if score > 0 else "Not found",
            source_page=None, formula_steps=steps,
            justification=steps, evidence_found=score > 0,
        )

    # ── LLM fallback ────────────────────────────────────────────────────────
    score, steps = apply_llm_score(criterion, response_text)
    return ScoreResult(
        criterion=criterion, score=score, max_marks=criterion.max_marks,
        extracted_value="LLM-assessed", source_page=None, formula_steps=steps,
        justification=steps, evidence_found=score > 0,
    )

# ---------------------------------------------------------------------------
# STAGE 4G — Multi-criterion revenue band handling (NHB style)
# ---------------------------------------------------------------------------

_NHB_REVENUE_BANDS = {
    # Average revenue from advisory (agri sector)
    "agri": [
        (5,   5,   "Rs 5.1 to 10 Cr → 5 marks"),
        (10,  10,  "Rs 10.1 to 15 Cr → 10 marks"),
        (15,  15,  "Above Rs 15 Cr → 15 marks"),
    ],
    # Overall average advisory revenue
    "overall": [
        (50,  1,   ">Rs 50 to 100 Cr → 1 mark"),
        (100, 2,   ">Rs 100 to 150 Cr → 2 marks"),
        (150, 3,   ">Rs 150 to 200 Cr → 3 marks"),
        (200, 5,   ">Rs 200 Cr → 5 marks"),
    ],
}

def apply_nhb_revenue_band(criteria_text: str, max_marks: int, value_cr: float) -> tuple[float, str]:
    """NHB-style revenue band scoring."""
    # Determine which band table to use
    if re.search(r"agri|horticulture|agriculture|allied", criteria_text, re.I):
        bands = _NHB_REVENUE_BANDS["agri"]
    else:
        bands = _NHB_REVENUE_BANDS["overall"]

    awarded = 0.0
    for threshold, marks, label in bands:
        if value_cr >= threshold:
            awarded = float(marks)

    awarded = _clamp(awarded, max_marks)
    return awarded, f"NHB-BAND: {value_cr} Cr → {awarded}/{max_marks}"

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_tq_evaluation(
    rfp_doc_name:      str,
    proposal_path:     str,
    proposal_doc_name: str,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Full TQ evaluation pipeline.

    Parameters
    ----------
    rfp_doc_name      : not used (criteria are extracted FROM the proposal)
    proposal_path     : absolute path to proposal PDF
    proposal_doc_name : unique identifier
    progress_callback : optional (step: str, pct: int) callable
    """

    def _prog(step: str, pct: int):
        if progress_callback:
            try: progress_callback(step, pct)
            except: pass
        print(f"[TQ] {pct:3d}%  {step}")

    _prog("Opening proposal PDF", 5)

    if not Path(proposal_path).exists():
        return _err(f"Proposal not found: {proposal_path}")

    doc = fitz.open(proposal_path)
    total_pages = len(doc)
    print(f"[TQ] Proposal: {Path(proposal_path).name} ({total_pages} pages)")

    # Stage 1: Find compliance matrix
    _prog("Finding compliance matrix pages", 10)
    matrix_pages = find_compliance_matrix_pages(doc)

    if not matrix_pages:
        doc.close()
        return _err("Could not locate compliance matrix in proposal")

    # Build full matrix text
    matrix_text = "\n".join(doc[p - 1].get_text() for p in matrix_pages)
    print(f"[TQ] Matrix text: {len(matrix_text)} chars from pages {matrix_pages}")

    # Stage 2: Parse criteria
    _prog("Parsing evaluation criteria", 20)
    criteria = parse_criteria_from_text(matrix_text)

    if not criteria:
        doc.close()
        return _err("No criteria found in compliance matrix")

    # Score live vs document criteria
    doc_criteria  = [c for c in criteria if not c.is_live]
    live_criteria = [c for c in criteria if c.is_live]
    doc_max       = sum(c.max_marks for c in doc_criteria)
    live_marks    = sum(c.max_marks for c in live_criteria)
    grand_total   = doc_max + live_marks

    print(f"[TQ] Found {len(criteria)} criteria: {len(doc_criteria)} document, "
          f"{len(live_criteria)} live")
    print(f"[TQ] doc_max={doc_max}, live_marks={live_marks}, grand_total={grand_total}")

    # Stage 3+4: Score each criterion
    _prog("Scoring criteria", 30)
    scores = []
    n = len(criteria)

    for i, criterion in enumerate(criteria):
        pct = 30 + int((i / max(n, 1)) * 60)
        _prog(f"Scoring: {criterion.parameter[:50]}", pct)

        # Get bidder's response text for this criterion
        response_text = extract_bidder_response(matrix_text, criterion)

        try:
            result = score_one_criterion(criterion, response_text, doc, matrix_text)
        except Exception as e:
            print(f"  [ERROR] {criterion.parameter}: {e}")
            result = ScoreResult(
                criterion=criterion, score=0.0, max_marks=criterion.max_marks,
                extracted_value=None, source_page=None, formula_steps=f"Error: {e}",
                justification=f"Scoring error: {e}", evidence_found=False,
                gaps=[str(e)],
            )

        sc = result.score
        print(f"  [{i+1:2d}/{n}] {criterion.parameter[:55]:55s} "
              f"{'--' if result.is_pending else f'{sc}/{criterion.max_marks}'}")
        scores.append(result)

    doc.close()

    # Stage 5: Aggregate
    _prog("Computing totals", 95)
    doc_scores   = [r for r in scores if not r.is_pending and r.score is not None]
    total_scored = round(sum(r.score for r in doc_scores), 1)
    total_pct    = round((total_scored / doc_max) * 100, 1) if doc_max > 0 else 0.0
    threshold    = 70.0

    qualified   = total_pct >= threshold
    qualification = {
        "threshold_pct": threshold,
        "achieved_pct":  total_pct,
        "passed":        qualified,
        "note": f"{'QUALIFIED' if qualified else 'NOT QUALIFIED'} — "
                f"{total_pct}% vs ≥{threshold}% required.",
    }

    print(f"\n[TQ] ─── Results ──────────────────────────────────────────")
    print(f"[TQ] Scored: {total_scored} / {doc_max} ({total_pct}%)")
    print(f"[TQ] Qualification: {'✅ QUALIFIED' if qualified else '❌ NOT QUALIFIED'}")
    print(f"[TQ] ──────────────────────────────────────────────────────\n")

    # Serialise scores to the existing API format
    api_scores = []
    for i, result in enumerate(scores):
        c = result.criterion
        api_scores.append({
            "item_code":                       c.item_code,
            "parameter":                       c.parameter[:295],
            "max_marks":                       c.max_marks,
            "criteria_text":                   c.criteria_text,
            "formula_hint":                    c.formula_type,
            "is_sub_item":                     c.is_sub_item,
            "parent_parameter":                c.parent_param,
            "score":                           result.score,
            "score_percentage":                (
                round((result.score / c.max_marks) * 100, 1) if result.score is not None and c.max_marks > 0 else None
            ),
            "extracted_value":                 result.extracted_value,
            "source_page":                     result.source_page,
            "scoring_steps":                   result.formula_steps,
            "justification":                   result.justification,
            "strengths":                       result.strengths,
            "gaps":                            result.gaps,
            "evidence_found":                  result.evidence_found,
            "evaluation_layer":                "live_assessment" if result.is_pending else "document",
            "requires_live_assessment":        result.is_pending,
            "requires_comparative_evaluation": False,
            "discrepancies":                   [],
        })

    return {
        "evaluation_title":            "Technical Evaluation",
        "grand_total_marks":           grand_total,
        "technical_document_max":      doc_max,
        "scoreable_total":             doc_max,
        "live_assessment_marks":       live_marks,
        "financial_marks":             0,
        "total_scored":                total_scored,
        "total_percentage":            total_pct,
        "final_score_formula":         None,
        "qualification_threshold":     threshold,
        "qualification":               qualification,
        "schema_valid":                True,
        "schema_warning":              None,
        "global_discrepancies":        [],
        "criteria_structure":          [asdict(c) for c in criteria],
        "scores":                      api_scores,
        "error":                       None,
    }

def ingest_proposal(proposal_path: str, proposal_doc_name: str) -> int:
    """Ingest proposal into vector store (compatibility shim)."""
    try:
        from core.parser import parse_document
        from core.vector_store import ingest_chunks
        chunks = parse_document(proposal_path)
        for c in chunks:
            c.doc_name = proposal_doc_name
        return ingest_chunks(chunks, doc_id=proposal_doc_name)
    except Exception as e:
        print(f"[TQ] Ingest skipped: {e}")
        return 0

def _err(msg: str) -> dict:
    print(f"[TQ] ERROR: {msg}")
    return {
        "evaluation_title": "Technical Evaluation",
        "grand_total_marks": 0, "technical_document_max": 0,
        "scoreable_total": 0, "live_assessment_marks": 0,
        "financial_marks": 0, "total_scored": 0, "total_percentage": 0.0,
        "final_score_formula": None, "qualification_threshold": 70.0,
        "qualification": {}, "schema_valid": False,
        "schema_warning": None, "global_discrepancies": [],
        "criteria_structure": [], "scores": [], "error": msg,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    proposals = {
        "udd":  "/mnt/user-data/uploads/UDD_UP_Proposal_3__1_.pdf",
        "nhb":  "/mnt/user-data/uploads/TechProposalNHBClusterDevelopmenGTILLP_Aug2020_Final.pdf",
        "gel":  "/mnt/user-data/uploads/GTBL_Proposal_GEL.pdf",
    }

    target = sys.argv[1] if len(sys.argv) > 1 else "udd"
    path   = proposals.get(target, proposals["udd"])

    print(f"\n{'='*70}")
    print(f"TQ Extractor v2 — Testing: {target.upper()}")
    print(f"File: {path}")
    print(f"{'='*70}\n")

    result = run_tq_evaluation("", path, target)

    print(f"\n{'='*70}")
    print(f"FINAL SCORE: {result['total_scored']} / {result['technical_document_max']}"
          f" ({result['total_percentage']}%)")
    print(f"Qualification: {result['qualification'].get('note','')}")
    print(f"{'='*70}")
    print(f"\nDetailed scores:")
    for s in result["scores"]:
        sc = s["score"]
        sc_str = "--pending--" if s["requires_live_assessment"] else f"{sc}/{s['max_marks']}"
        print(f"  [{s['item_code']:>2}] {s['parameter'][:55]:55s} {sc_str:>12}  [{s['formula_hint']}]")
        if s.get("extracted_value"):
            print(f"       └─ {s['extracted_value']}")