"""
core/tq_extractor.py  —  v19  Cache-First Architecture
=======================================================

KEY CHANGES FROM v18
─────────────────────
FIX 1 — CACHING (eliminates non-determinism)
  The #1 failure mode: same RFP extracted 4 times → 4 different criterion sets.
  v19 caches to rfp_cache/<hash>.json after the FIRST successful extraction.
  Subsequent runs load from cache → always identical criteria + bands.
  Cache invalidates only when the PDF file changes (hash changes).

FIX 2 — BAND PRE-COMPUTATION
  v18 re-asked the LLM for scoring bands at scoring time (once per criterion).
  v19 extracts all bands ONCE during extraction, stores them in the cache.
  Scoring becomes fully deterministic (no LLM calls for formula structure).

FIX 3 — IMPROVED EXTRACTION PROMPT
  The 30-mark "Other Support Activities" criterion was missed ~50% of the time.
  Root cause: it appears as row 6 in the RFP table, and the LLM sometimes
  confuses the 30-mark Technical Presentation with the 30-mark document criterion.
  Fix: explicit instruction to count rows, verify sum = doc_total before returning.

FIX 4 — THREE-PASS EXTRACTION
  Pass 1: Normal extraction
  Pass 2: If sum mismatch > 2, targeted repair with "missing N marks" context
  Pass 3: If still wrong, page-by-page scan for any missed "N marks" rows

PUBLIC API UNCHANGED
  extract_marking_table(), ingest_proposal(), run_tq_evaluation() — same signatures
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from core.llm_client import call_llm, extract_json
from core.rfp_cache import load_cache, save_cache, precompute_bands
from core.parser import parse_document
from core.vector_store import ingest_chunks

try:
    from core.tq_extractor_patch_2 import extract_with_llm_safe as _patch2_extract
    _PATCH2_OK = True
except ImportError:
    _PATCH2_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TQ_UPLOAD_DIR        = Path("./tq_uploads")
TQ_UPLOAD_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_CHARS    = 90_000
MAX_SECTION_SPAN     = 30
PAGE_SCORE_THRESHOLD = 4
HOT_THRESHOLD        = 6
_DB_PARAM_MAX        = 295

_RFP_SEARCH_DIRS = [
    Path("./uploads"),
    Path("./tq_uploads"),
    Path("./rfp_uploads"),
    Path("."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Regex helpers (unchanged from v18)
# ─────────────────────────────────────────────────────────────────────────────

_MARKS_SIGNAL = re.compile(
    r"(\d+\s*marks?\b|max(?:imum)?\.?\s*marks?|marks?\s*[:=]\s*\d+|\d+\s*points?)",
    re.IGNORECASE,
)
_PARAM_SIGNAL = re.compile(
    r"(turnover|experience|qualification|competence|methodology|"
    r"personnel|manpower|professional|net\s*worth|revenue|certification|"
    r"technical\s+support|project\s+management|skill\s+development|"
    r"tsa\b|pmu\b|pmc\b|stsa\b)",
    re.IGNORECASE,
)
_SCORING_TABLE_HEADER = re.compile(
    r"(s[\.\s]*no\.?|serial\s+no\.?).{0,300}?"
    r"(parameter(\s+name)?|criterion|criteria|particulars|description).{0,300}?"
    r"(max(?:imum)?\.?\s*marks?|full\s+marks?|marks\s+criteria)",
    re.IGNORECASE | re.DOTALL,
)
_CONTRACT_SIGNAL = re.compile(
    r"(indemnity|arbitration|commencement\s+of\s+work|force\s+majeure|"
    r"contract\s+termination|penalty\s+clause)",
    re.IGNORECASE,
)
_TOR_ACTION_PREFIXES = re.compile(
    r"^(assist\b|monitor\b|submit\b|prepare\b|provide\b|coordinate\b|"
    r"ensure\b|review\b|facilitate\b|he\s*/\s*she\b|the\s+consultant\s+shall)",
    re.IGNORECASE,
)
_EVAL_SECTION_KW = re.compile(
    r"(criteria\s+for\s+(technical\s+)?evaluation|evaluation\s+(of\s+)?criteria|"
    r"evaluation\s+of\s+technical\s+bid|technical\s+bid\s+eval(uation)?|"
    r"scoring\s+criteria|technical\s+evaluation\s*$)",
    re.IGNORECASE,
)
_NEXT_SECTION_KW = re.compile(
    r"(short.?list(ing)?|evaluation\s+of\s+financial|financial\s+bid\s+eval|"
    r"combined\s+and\s+final|general\s+conditions|special\s+conditions|"
    r"fraud\s+and\s+corrupt|tender\s+methodology|commercial\s+bid)",
    re.IGNORECASE,
)
_SKIP_PATTERNS = re.compile(
    r"(^presentation$|^interview$|viva\b|^demo$|^panel$|financial\s+bid|"
    r"price\s+bid|\bL1\b|commercial\s+bid|indemnity|arbitration|"
    r"combined\s+and\s+final|appreciation\s+and\s+response|"
    r"evaluation\s+of\s+financial|opening\s+of.*financial)",
    re.IGNORECASE | re.VERBOSE,
)
_LIVE_PATTERNS = re.compile(
    r"(presentation\b|interview\b|viva\b|panel\s+discussion|"
    r"technical\s+presentation|virtual\s+presentation)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: str, n: int = _DB_PARAM_MAX) -> str:
    return (text[:n - 3] + "...") if text and len(text) > n else (text or "")


def _find_rfp_pdf(rfp_doc_name: str) -> Optional[Path]:
    for d in _RFP_SEARCH_DIRS:
        p = d / rfp_doc_name
        if p.exists():
            return p
    return None


def _is_live_assessment(parameter: str) -> bool:
    return bool(_LIVE_PATTERNS.search(parameter or ""))


def _validate_criterion(c: dict) -> bool:
    name  = (c.get("parameter") or "").strip()
    marks = c.get("max_marks", 0)
    if not name or not marks or marks < 1:
        return False
    if _SKIP_PATTERNS.search(name) or _is_live_assessment(name):
        print(f"[TQ] Skipping (live/skip): {name[:60]}")
        return False
    return True


def _zero_result(reason: str) -> dict:
    return {
        "score": 0.0, "extracted_value": None, "source_page": None,
        "scoring_steps": reason, "justification": reason,
        "strengths": [], "gaps": [reason], "evidence_found": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _fitz_extract(pdf_path: str, start: int, end: int) -> Optional[str]:
    try:
        import fitz
    except ImportError:
        return None
    try:
        doc   = fitz.open(pdf_path)
        parts = []
        for pg_idx in range(start - 1, min(end, len(doc))):
            txt = doc[pg_idx].get_text("text")
            parts.append(txt)
        doc.close()
        return "\f".join(parts) if parts else None
    except Exception as e:
        print(f"[TQ] fitz extract error: {e}")
        return None


def _pdftotext(pdf_path: str, start: int, end: int, layout: bool = True) -> Optional[str]:
    result = _fitz_extract(pdf_path, start, end)
    if result and result.strip():
        return result
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd += ["-f", str(start), "-l", str(end), pdf_path, "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.stdout if r.returncode == 0 and r.stdout.strip() else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"[TQ] pdftotext error: {e}")
        return None


def _extract_trailing_page(line: str) -> Optional[int]:
    m = re.search(r'\b(\d{1,3})\s*$', line.rstrip())
    return int(m.group(1)) if m else None


def _parse_toc(text: str) -> tuple[Optional[int], Optional[int]]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if _EVAL_SECTION_KW.search(line):
            pg = _extract_trailing_page(line)
            if pg and 5 <= pg <= 250:
                end_pg = None
                for j in range(i + 1, min(i + 20, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        continue
                    if _NEXT_SECTION_KW.search(nxt):
                        ep = _extract_trailing_page(nxt)
                        if ep and ep >= pg:
                            end_pg = ep
                        break
                    ep = _extract_trailing_page(nxt)
                    if ep and ep > pg and re.match(r'^[\s\d\.]+', nxt):
                        end_pg = ep
                        break
                print(f"[TQ] TOC: eval section p{pg} → p{end_pg}")
                return pg, end_pg
    print("[TQ] TOC: eval section not found")
    return None, None


def _score_page(page_text: str, page_no: int,
                toc_lo: Optional[int], toc_hi: Optional[int]) -> float:
    score = 0.0
    if toc_lo is not None and toc_hi is not None:
        if toc_lo <= page_no <= toc_hi:
            score += 10
    if _SCORING_TABLE_HEADER.search(page_text):
        score += 8
    mhits = len(_MARKS_SIGNAL.findall(page_text))
    if mhits >= 4:   score += 5
    elif mhits >= 1: score += 3
    phits = len(_PARAM_SIGNAL.findall(page_text))
    if phits >= 3:   score += 3
    elif phits >= 1: score += 1
    cn = len(_CONTRACT_SIGNAL.findall(page_text))
    if cn > 2 and mhits == 0:
        score -= 5
    tor_lines = sum(1 for ln in page_text.splitlines()
                    if _TOR_ACTION_PREFIXES.match(ln.strip()[:60]))
    if tor_lines > 5:
        score -= 3
    return score


def _find_eval_cluster(pdf_path: str) -> tuple[list[int], Optional[int], Optional[int]]:
    toc_text   = _pdftotext(pdf_path, 1, 30) or ""
    toc_start, toc_end = _parse_toc(toc_text)
    if toc_start:
        scan_start = max(1, toc_start - 2)
        scan_end   = (toc_end + 5) if toc_end else (toc_start + MAX_SECTION_SPAN + 5)
    else:
        scan_start, scan_end = 1, 80

    raw_text = _pdftotext(pdf_path, scan_start, scan_end) or ""
    pages    = raw_text.split("\f")

    toc_lo = (toc_start - 1) if toc_start else None
    toc_hi = ((toc_end + 1) if toc_end else None) if toc_start else None

    page_scores: dict[int, float] = {}
    for i, pt in enumerate(pages):
        pg = scan_start + i
        if pg > scan_end or not pt.strip():
            continue
        page_scores[pg] = _score_page(pt, pg, toc_lo, toc_hi)

    hot = {pg for pg, sc in page_scores.items() if sc >= HOT_THRESHOLD}
    final: dict[int, float] = {}
    for pg, sc in page_scores.items():
        boost = sum(2.0 for adj in [pg - 1, pg + 1] if adj in hot)
        final[pg] = sc + boost

    top = sorted(final.items(), key=lambda x: -x[1])[:12]
    print("[TQ] Top page scores:")
    for pg, sc in top:
        print(f"     p{pg:3d}  score={sc:5.1f}")

    selected = sorted(pg for pg, sc in final.items() if sc >= PAGE_SCORE_THRESHOLD)
    if not selected:
        if toc_start:
            cluster = list(range(toc_start, (toc_end or toc_start + 6) + 1))
            print(f"[TQ] No cluster scored → using TOC range: {cluster}")
            return cluster, toc_start, toc_end
        return [], toc_start, toc_end

    clusters: list[list[int]] = [[selected[0]]]
    for pg in selected[1:]:
        if pg - clusters[-1][-1] <= 3:
            clusters[-1].append(pg)
        else:
            clusters.append([pg])

    def weight(c: list[int]) -> float:
        vals = [final[p] for p in c]
        return len(c) * (sum(vals) / len(vals))

    if toc_start:
        top2 = sorted(clusters, key=weight, reverse=True)[:2]
        best = min(top2, key=lambda c: min(abs(p - toc_start) for p in c))
    else:
        best = max(clusters, key=weight)

    if best[-1] - best[0] > MAX_SECTION_SPAN:
        best = [p for p in best if p <= best[0] + MAX_SECTION_SPAN]

    print(f"[TQ] Cluster: {best}")
    return best, toc_start, toc_end


def _extract_eval_text(pdf_path: str, cluster: list[int]) -> str:
    if not cluster:
        return ""
    start, end = min(cluster), max(cluster)
    raw = _pdftotext(pdf_path, start, end, layout=True)
    if not raw:
        raw = _pdftotext(pdf_path, start, end, layout=False) or ""
    if not raw:
        return ""
    parts = []
    for i, pt in enumerate(raw.split("\f")):
        pg = start + i
        if pg > end:
            break
        if pt.strip():
            parts.append(f"[PAGE {pg}]\n{pt.strip()}")
    text = "\n\n".join(parts)
    print(f"[TQ] Eval text: {len(text)} chars from pages {start}-{end}")
    return text[:MAX_CONTEXT_CHARS]


# ─────────────────────────────────────────────────────────────────────────────
# Improved Extraction Prompt (FIX 3)
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are analysing the Technical Bid Evaluation section of a government RFP.
[PAGE N] markers indicate page breaks.

════════════════════════════════════════════════════════════════
CRITICAL INSTRUCTIONS — READ BEFORE EXTRACTING:

1. Count EVERY numbered row in the scoring table (S.No 1, 2, 3, ...).
   Do NOT stop at the first sub-criteria — continue to the very last row.

2. The sum of ALL top-level max_marks MUST equal doc_total (= grand_total − live_marks).
   If your sum is less, YOU HAVE MISSED A ROW.  Search the text again.

3. The "Technical Presentation" / "Technical Presentation by Bidder" row is LIVE ASSESSMENT.
   Put it in live_assessment_marks.  Do NOT include it in criteria[].
   However, if there is ANOTHER 30-mark (or similar) document-scoreable row (like
   "Other Support Activities", "Additional Experience", "Support Experience", etc.),
   that IS a document criterion — include it in criteria[].

4. search_keywords: include ALL acronyms, scheme names, role titles, and exact phrases
   from the RFP that a compliant bidder would use in their proposal.

5. Verify: sum_check = sum of top-level max_marks = doc_total.
════════════════════════════════════════════════════════════════

FORMULA TYPES:
  BAND      Discrete buckets → marks  (e.g. "100-200 Crs: 5 marks")
  STEP      Base + increment per unit
  PER_UNIT  Rate × count, capped
  QUAL      CV/team composition criteria
  BINARY    Yes/No (e.g. "one or more project: 10 marks")
  LLM       Complex/narrative

SUB-CRITERIA RULES:
  If S.No 5 has rows 5a and 5b each with their own marks:
    - Extract each as a sub-criterion with its own max_marks
    - Parent max_marks = sum of sub-criteria
    - Both parent AND sub-criteria go in criteria[]

RFP TEXT:
{text}

Return ONLY valid JSON — no markdown, no preamble:
{{
  "grand_total": <integer>,
  "live_assessment_marks": <integer or 0>,
  "live_assessment_label": "<e.g. Technical Presentations by Bidder>",
  "doc_total": <grand_total - live_assessment_marks>,
  "qualification_threshold_pct": <number or null>,
  "evaluation_title": "<section heading>",
  "criteria": [
    {{
      "item_code": "1",
      "parameter": "<short name, < 60 chars>",
      "max_marks": <integer>,
      "formula_type": "BAND|STEP|PER_UNIT|QUAL|BINARY|LLM",
      "criteria_text": "<VERBATIM scoring rules from RFP>",
      "search_keywords": ["<keyword1>", "<keyword2>", "..."],
      "sub_criteria": [
        {{
          "item_code": "1a",
          "parameter": "<sub name>",
          "max_marks": <integer>,
          "formula_type": "BAND|STEP|PER_UNIT|QUAL|BINARY|LLM",
          "criteria_text": "<VERBATIM sub-criterion rules>",
          "search_keywords": ["<keyword1>", "..."]
        }}
      ]
    }}
  ],
  "sum_check": <must equal doc_total>,
  "skipped_rows": ["<name: reason>"],
  "extraction_notes": "<any issues>"
}}
"""

_REPAIR_PROMPT = """\
You previously extracted technical evaluation criteria from an RFP.
Document criteria should sum to {doc_total} marks but actually sum to {actual_sum}.
You are MISSING {delta} marks.

THE MOST COMMON CAUSE: There is a row in the scoring table that was not extracted.
Often this is a row like "Other Support Activities", "Additional Experience",
"Relevant Experience", or any row near the bottom of the table.
It may be worth {delta} marks.

Look specifically for:
  1. Any numbered S.No that was skipped
  2. Any row near the bottom of the table (after S.No 5)
  3. Any row describing IEC / support / advisory activities with marks attached

Previous extraction (may be incomplete):
{previous_json}

RFP text:
{text}

Return ONLY corrected, complete JSON.
sum_check MUST equal {doc_total}.
Each criterion MUST include search_keywords[].
"""

_SCAN_PROMPT = """\
A scoring table in this RFP text is missing {delta} marks worth of criteria.
The criteria extracted so far sum to {actual_sum} but should sum to {doc_total}.

Scan the RFP text below for any row/line that:
  - Contains a number of marks (e.g. "30 marks", "Max Marks: 30")
  - Is NOT already in the extracted criteria listed below
  - Is NOT a live presentation / technical presentation row

Already extracted parameters: {extracted_params}

RFP TEXT (focus on any scoring tables you see):
{text}

Return ONLY valid JSON listing ONLY the NEW/MISSING criteria:
{{
  "missing_criteria": [
    {{
      "item_code": "6",
      "parameter": "<short name>",
      "max_marks": <integer>,
      "formula_type": "BAND|BINARY|LLM",
      "criteria_text": "<verbatim RFP text>",
      "search_keywords": ["<kw1>", "<kw2>"]
    }}
  ]
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Extraction (three-pass, FIX 4)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_with_llm(eval_text: str) -> tuple[int, int, str, float, list]:
    """Three-pass extraction: extract → repair → scan."""

    # ── Pass 1: Normal extraction ─────────────────────────────────────────
    prompt = _EXTRACTION_PROMPT.format(text=eval_text)
    print(f"[TQ] Extraction prompt: {len(prompt):,} chars → sending to LLM")

    raw    = call_llm(prompt, label="tq-extract-v19")
    parsed = extract_json(raw)

    if not parsed:
        print("[TQ] Extraction: LLM returned no parseable JSON")
        return 100, 0, "", 70.0, []

    grand_total = int(parsed.get("grand_total") or 100)
    live_marks  = int(parsed.get("live_assessment_marks") or 0)
    live_label  = str(parsed.get("live_assessment_label") or "")
    doc_total   = int(parsed.get("doc_total") or (grand_total - live_marks))
    threshold   = float(parsed.get("qualification_threshold_pct") or 70.0)

    if live_marks:
        print(f"[TQ] Live assessment: {live_marks} marks ({live_label})")

    raw_criteria = parsed.get("criteria", [])
    criteria     = [c for c in raw_criteria if _validate_criterion(c)]
    actual_sum   = sum(c.get("max_marks", 0) for c in criteria)

    print(f"[TQ] Pass 1: grand={grand_total}, live={live_marks}, "
          f"doc_total={doc_total}, criteria={len(criteria)}, sum={actual_sum}")

    # ── Pass 2: Repair if sum mismatch > 2 ───────────────────────────────
    if abs(actual_sum - doc_total) > 2:
        delta = doc_total - actual_sum
        print(f"[TQ] Pass 2: Sum mismatch — repairing (missing {delta} marks)")
        rp = _REPAIR_PROMPT.format(
            doc_total    = doc_total,
            actual_sum   = actual_sum,
            delta        = abs(delta),
            previous_json= json.dumps(parsed, indent=2)[:5000],
            text         = eval_text[:6000],
        )
        raw2    = call_llm(rp, label="tq-repair-v19")
        parsed2 = extract_json(raw2)
        if parsed2 and parsed2.get("criteria"):
            crit2   = [c for c in parsed2["criteria"] if _validate_criterion(c)]
            actual2 = sum(c.get("max_marks", 0) for c in crit2)
            print(f"[TQ] Pass 2 result: {len(crit2)} criteria, sum={actual2}")
            if abs(actual2 - doc_total) < abs(actual_sum - doc_total):
                criteria   = crit2
                actual_sum = actual2

    # ── Pass 3: Targeted scan for missing rows ────────────────────────────
    if abs(actual_sum - doc_total) > 2:
        delta = doc_total - actual_sum
        print(f"[TQ] Pass 3: Still {delta} marks missing — scanning for missed rows")
        extracted_params = [c.get("parameter", "") for c in criteria]
        sp = _SCAN_PROMPT.format(
            delta            = abs(delta),
            actual_sum       = actual_sum,
            doc_total        = doc_total,
            extracted_params = json.dumps(extracted_params),
            text             = eval_text[:8000],
        )
        raw3    = call_llm(sp, label="tq-scan-v19")
        parsed3 = extract_json(raw3)
        if parsed3 and parsed3.get("missing_criteria"):
            new_crit = [c for c in parsed3["missing_criteria"] if _validate_criterion(c)]
            if new_crit:
                criteria   = criteria + new_crit
                actual_sum = sum(c.get("max_marks", 0) for c in criteria)
                print(f"[TQ] Pass 3: Added {len(new_crit)} criteria → sum={actual_sum}")

    return grand_total, live_marks, live_label, threshold, criteria


def _flatten_criteria(criteria: list) -> list:
    """Flatten parent+sub_criteria into a flat list."""
    flat: list = []
    for c in criteria:
        subs = c.get("sub_criteria") or []
        if subs:
            flat.append({**c, "sub_criteria": [], "is_parent": True, "is_sub_item": False})
            for s in subs:
                if _validate_criterion(s):
                    flat.append({
                        **s,
                        "sub_criteria":     [],
                        "is_parent":        False,
                        "is_sub_item":      True,
                        "parent_parameter": c.get("parameter", ""),
                    })
        else:
            flat.append({**c, "sub_criteria": [], "is_parent": False, "is_sub_item": False})
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Public extraction entry point  (CACHE-FIRST — FIX 1 + 2)
# ─────────────────────────────────────────────────────────────────────────────

def extract_marking_table(rfp_doc_name: str, force_refresh: bool = False) -> dict:
    """
    Full extraction pipeline — cache-first.

    If a valid cache exists for this PDF, returns cached criteria immediately.
    Only runs the LLM pipeline on first call (or if force_refresh=True).

    Args:
        rfp_doc_name:   filename of the RFP PDF (searched in _RFP_SEARCH_DIRS)
        force_refresh:  True → ignore cache, re-extract and re-save
    """
    print(f"[TQ] Extracting marking table from: {rfp_doc_name}")

    pdf_path = _find_rfp_pdf(rfp_doc_name)
    if not pdf_path:
        return {
            "criteria": [], "grand_total_marks": 0,
            "error": f"RFP PDF not found: {rfp_doc_name}",
            "context_source": "none",
        }

    print(f"[TQ] Found RFP PDF: {pdf_path}")

    # ── Cache lookup ───────────────────────────────────────────────────────
    if not force_refresh:
        cached = load_cache(str(pdf_path))
        if cached:
            flat = cached["criteria"]
            return {
                "evaluation_title":            "Technical Evaluation",
                "grand_total_marks":           cached["grand_total"],
                "live_assessment_marks":       cached["live_marks"],
                "live_assessment_label":       cached["live_label"],
                "qualification_threshold_pct": cached["threshold"],
                "criteria":                    flat,
                "doc_max":                     cached["doc_total"],
                "schema_warning":              None,
                "context_source":              cached.get("context_source", "cache"),
                "bands":                       cached.get("bands", {}),
                "error":                       None,
                "_from_cache":                 True,
            }

    # ── Fresh extraction ───────────────────────────────────────────────────
    cluster, toc_start, toc_end = _find_eval_cluster(str(pdf_path))

    if not cluster:
        if toc_start:
            cluster = list(range(toc_start, (toc_end or toc_start + 8) + 1))
        else:
            return {
                "criteria": [], "grand_total_marks": 0,
                "error": "Could not identify evaluation section pages.",
                "context_source": "none",
            }

    eval_text = _extract_eval_text(str(pdf_path), cluster)
    if not eval_text:
        return {
            "criteria": [], "grand_total_marks": 0,
            "error": "PDF text extraction returned empty.",
            "context_source": f"p{cluster[0]}-p{cluster[-1]}",
        }

    grand_total, live_marks, live_label, threshold, criteria = \
        _extract_with_llm(eval_text)

    if not criteria:
        return {
            "criteria": [], "grand_total_marks": grand_total,
            "live_assessment_marks": live_marks,
            "live_assessment_label": live_label,
            "error": "LLM returned no valid criteria.",
            "context_source": f"p{cluster[0]}-p{cluster[-1]}",
        }

    flat    = _flatten_criteria(criteria)
    doc_max = sum(c.get("max_marks", 0) for c in criteria)
    exp_doc = grand_total - live_marks

    schema_warning = None
    if abs(doc_max - exp_doc) > 2:
        schema_warning = (
            f"Criteria sum to {doc_max} but expected {exp_doc} "
            f"(grand {grand_total} – live {live_marks})."
        )

    n_score = len([c for c in flat if not c.get("is_parent")])
    print(f"[TQ] Final: {n_score} scoreable | sum={doc_max} | "
          f"expected={exp_doc} | grand={grand_total}"
          + (f" | live={live_marks} ({live_label})" if live_marks else ""))
    for c in flat:
        pfx = "  " if c.get("is_sub_item") else ""
        tag = " [parent]" if c.get("is_parent") else f" [{c.get('formula_type','?')}]"
        kws = c.get("search_keywords", [])
        print(f"  {pfx}[{str(c.get('item_code','?')):4s}] "
              f"{c['parameter'][:50]:50s}  {c['max_marks']:3d}{tag}  "
              f"keywords={kws[:3]}")

    result = {
        "evaluation_title":            "Technical Evaluation",
        "grand_total_marks":           grand_total,
        "live_assessment_marks":       live_marks,
        "live_assessment_label":       live_label,
        "qualification_threshold_pct": threshold,
        "criteria":                    flat,
        "doc_max":                     doc_max,
        "schema_warning":              schema_warning,
        "context_source":              f"p{cluster[0]}-p{cluster[-1]}",
        "error":                       None,
        "_from_cache":                 False,
    }

    # ── Pre-compute bands and save to cache (FIX 2) ────────────────────────
    if not schema_warning:
        try:
            bands = precompute_bands(flat)
            result["bands"] = bands
            save_cache(str(pdf_path), result, bands)
        except Exception as e:
            print(f"[TQ] Band pre-computation / cache error (non-fatal): {e}")
            result["bands"] = {}
    else:
        print(f"[TQ] Skipping cache save due to schema_warning: {schema_warning}")
        result["bands"] = {}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Proposal ingestion (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_proposal(proposal_path: str, proposal_doc_name: str) -> int:
    print(f"[TQ] Ingesting proposal: {proposal_path}")
    try:
        chunks = parse_document(proposal_path)
        for chunk in chunks:
            chunk.doc_name = proposal_doc_name
            chunk.chunk_id = (f"{proposal_doc_name}_{chunk.page_no}_"
                              f"{chunk.chunk_id.split('_')[-1]}")
        count = ingest_chunks(chunks, doc_id=proposal_doc_name)
        print(f"[TQ] Proposal ingested: {count} chunks")
        return count
    except Exception as e:
        print(f"[TQ] Proposal ingestion error (non-fatal): {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator (public API — unchanged signatures)
# ─────────────────────────────────────────────────────────────────────────────

def run_tq_evaluation(
    rfp_doc_name:      str,
    proposal_path:     str,
    proposal_doc_name: str,
    progress_callback: Optional[Callable] = None,
    force_rfp_refresh: bool = False,
) -> dict:
    """
    Full TQ evaluation pipeline.
    force_rfp_refresh=True → ignore cache, re-extract RFP criteria.
    """

    def _prog(step: str, pct: int):
        if progress_callback:
            try:
                progress_callback(step, pct)
            except Exception:
                pass
        print(f"[TQ] {pct:3d}% -- {step}")

    _prog("Reading RFP marking table", 10)
    table      = extract_marking_table(rfp_doc_name, force_refresh=force_rfp_refresh)
    criteria   = table.get("criteria", [])
    live_marks = int(table.get("live_assessment_marks") or 0)
    live_label = str(table.get("live_assessment_label") or "")
    # Pass pre-computed bands to scorer
    rfp_bands  = table.get("bands", {})

    if not criteria:
        print(f"[TQ] No criteria — error: {table.get('error')}")
        return {
            "evaluation_title":       "Technical Evaluation",
            "grand_total_marks":      table.get("grand_total_marks", 0),
            "technical_document_max": 0,
            "scoreable_total":        0,
            "live_assessment_marks":  live_marks,
            "live_assessment_label":  live_label,
            "financial_marks":        0,
            "total_scored":           0,
            "total_percentage":       0.0,
            "final_score_formula":    None,
            "qualification_threshold": table.get("qualification_threshold_pct"),
            "qualification":          {},
            "schema_valid":           False,
            "schema_warning":         table.get("schema_warning"),
            "criteria_structure":     [],
            "scores":                 [],
            "error":                  table.get("error", "No criteria extracted."),
        }

    scoreable   = [c for c in criteria if not c.get("is_parent")]
    parents     = [c for c in criteria if c.get("is_parent")]
    grand_total = table.get("grand_total_marks", 100)
    threshold   = table.get("qualification_threshold_pct", 70.0)
    doc_max     = sum(c.get("max_marks", 0) for c in scoreable)
    from_cache  = table.get("_from_cache", False)

    live_msg = (f"; live assessment: {live_marks} marks ({live_label})"
                if live_marks else "")
    cache_msg = " [from cache]" if from_cache else " [freshly extracted]"
    _prog(f"Found {len(scoreable)} scoreable criteria ({doc_max} marks){live_msg}{cache_msg}", 15)
    _prog("Ingesting proposal into vector store", 20)
    ingest_proposal(proposal_path, proposal_doc_name)

    # ── Warm proposal analysis cache (1 bulk LLM call total) ──────────────
    _prog("Analyzing proposal (bulk extraction)", 23)
    try:
        from core.tq_scorer import warm_analysis_cache
        warm_analysis_cache(proposal_path, scoreable)
    except Exception as e:
        print(f"[TQ] Analysis warm error (non-fatal): {e}")
    _prog("Scoring criteria against proposal", 28)
    # ──────────────────────────────────────────────────────────────────────

    # Import scorer — pass rfp_bands so it can use them without extra LLM calls
    try:
        from core.tq_scorer import score_criterion
    except ImportError:
        from core.tq_scorer_v2 import score_criterion

    scores: list[dict] = []
    n = len(scoreable)

    for i, criterion in enumerate(scoreable):
        pct = 28 + int((i / max(n, 1)) * 65)
        _prog(f"Scoring: {criterion['parameter'][:55]}", pct)

        # Inject cached bands into criterion so scorer doesn't need to re-fetch
        if rfp_bands and criterion.get("parameter") in rfp_bands:
            criterion = {**criterion, "_cached_bands": rfp_bands[criterion["parameter"]]}

        try:
            result = score_criterion(criterion, proposal_path, all_criteria=scoreable)
        except Exception as e:
            print(f"[TQ] Error scoring '{criterion['parameter']}': {e}")
            import traceback; traceback.print_exc()
            result = _zero_result(str(e))

        s  = result.get("score", 0)
        pg = f"(p.{result['source_page']})" if result.get("source_page") else ""
        print(f"  [{i+1}/{n}] {criterion['parameter'][:55]:55s} "
              f"→ {s}/{criterion['max_marks']} {pg}")

        scores.append({
            "item_code":                       criterion.get("item_code", str(i + 1)),
            "parameter":                       _truncate(criterion["parameter"]),
            "max_marks":                       criterion["max_marks"],
            "criteria_text":                   criterion.get("criteria_text", ""),
            "is_sub_item":                     bool(criterion.get("is_sub_item", False)),
            "parent_parameter":                criterion.get("parent_parameter", ""),
            "evaluation_layer":                "document",
            "requires_live_assessment":        False,
            "requires_comparative_evaluation": False,
            **result,
        })

    # Parent display rows
    for p in parents:
        child_scores = [s for s in scores
                        if s.get("parent_parameter") == p.get("parameter")]
        p_score = sum(c.get("score", 0) or 0 for c in child_scores)
        scores.append({
            "item_code":            p.get("item_code", ""),
            "parameter":            _truncate(p["parameter"]),
            "max_marks":            p["max_marks"],
            "criteria_text":        p.get("criteria_text", ""),
            "is_sub_item":          False,
            "parent_parameter":     "",
            "evaluation_layer":     "document",
            "requires_live_assessment": False,
            "requires_comparative_evaluation": False,
            "score":                p_score,
            "extracted_value":      "Sum of sub-criteria",
            "source_page":          None,
            "scoring_steps":        f"Parent = sum of sub-criteria = {p_score}",
            "justification":        f"Score {p_score}/{p['max_marks']} (sum)",
            "strengths":            [],
            "gaps":                 [],
            "evidence_found":       p_score > 0,
        })

    # Live assessment placeholder
    if live_marks > 0:
        scores.append({
            "item_code":            "L1",
            "parameter":            _truncate(live_label or "Technical Presentation"),
            "max_marks":            live_marks,
            "criteria_text":        "Live panel presentation — scored by committee",
            "is_sub_item":          False,
            "parent_parameter":     "",
            "evaluation_layer":     "live_assessment",
            "requires_live_assessment": True,
            "requires_comparative_evaluation": False,
            "score":                None,
            "extracted_value":      "Pending panel evaluation",
            "source_page":          None,
            "scoring_steps":        "Live assessment — cannot score from document",
            "justification":        "Pending live presentation evaluation",
            "strengths":            [],
            "gaps":                 ["Live panel evaluation required"],
            "evidence_found":       False,
        })

    _prog("Computing totals", 96)

    leaf_scores  = [s for s in scores
                    if s.get("evaluation_layer") == "document"
                    and s.get("extracted_value") != "Sum of sub-criteria"
                    and not s.get("is_parent")]
    total_scored = round(sum(s.get("score") or 0 for s in leaf_scores), 1)
    total_pct    = round((total_scored / doc_max) * 100, 1) if doc_max > 0 else 0.0

    qualification: dict = {}
    if threshold:
        min_doc = round(threshold / 100.0 * doc_max, 1)
        passed  = total_scored >= min_doc
        qualification = {
            "threshold_pct":       float(threshold),
            "min_doc_marks":       min_doc,
            "achieved_doc_marks":  total_scored,
            "achieved_pct":        total_pct,
            "passed":              passed,
            "financial_bid_opens": passed,
            "note": (
                f"Qualified (document: {total_scored}/{doc_max}). "
                f"Live assessment ({live_marks} marks) pending."
                if passed else
                f"Not qualified — {total_scored} < {min_doc} required doc marks."
            ),
        }
        print(f"[TQ] Gate: {'QUALIFIED' if passed else 'NOT QUALIFIED'} "
              f"({total_scored}/{doc_max} doc, {total_pct}%)")

    print(f"[TQ] Result: {total_scored}/{doc_max}  ({total_pct}%)"
          + (f"  |  Live pending: {live_marks} marks" if live_marks else ""))

    _prog("Done", 100)

    return {
        "evaluation_title":        table.get("evaluation_title", "Technical Evaluation"),
        "grand_total_marks":       grand_total,
        "technical_document_max":  doc_max,
        "scoreable_total":         doc_max,
        "live_assessment_marks":   live_marks,
        "live_assessment_label":   live_label,
        "financial_marks":         0,
        "total_scored":            total_scored,
        "total_percentage":        total_pct,
        "final_score_formula":     None,
        "qualification_threshold": threshold,
        "qualification":           qualification,
        "schema_valid":            table.get("schema_warning") is None,
        "schema_warning":          table.get("schema_warning"),
        "criteria_structure":      criteria,
        "scores":                  scores,
        "error":                   table.get("error"),
        "from_cache":              from_cache,
    }