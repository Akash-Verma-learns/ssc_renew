"""
core/rfp_cache.py  —  v1  Deterministic RFP Extraction Cache
=============================================================

WHY THIS EXISTS
───────────────
The LLM is non-deterministic.  The same RFP PDF, run four times, produced:
  - Run 1: sum=100 (all criteria found)
  - Run 2: sum=70  (30-mark criterion missed)
  - Run 3: sum=100 (all criteria found, different name for criterion 6)
  - Run 4: sum=70  (multiple criteria missed)

This module solves that by caching the extraction result to disk.
The cache key is sha256(pdf_bytes)[:16] — changes only when the PDF changes.

WHAT IS CACHED
──────────────
  {
    "rfp_hash":     "abc123...",
    "rfp_filename": "my_rfp.pdf",
    "cached_at":    "2024-10-28T12:00:00",
    "grand_total":  130,
    "live_marks":   30,
    "live_label":   "Technical Presentations",
    "doc_total":    100,
    "threshold":    70.0,
    "criteria":     [...],   # full flat criteria list with search_keywords
    "bands":        {        # pre-computed scoring bands per parameter
      "Financial Turnover": [{"min": 100, "max": 200, "score": 5}, ...],
      "Area of Experience": [{"min": 3, "max": 5, "score": 4}, ...],
    }
  }

BANDS PRE-COMPUTATION
──────────────────────
At cache-write time, we extract ALL scoring bands from criteria_text in one
LLM call per criterion.  At scoring time, bands are read from disk — zero
LLM calls needed for formula structure.

CACHE LOCATION
──────────────
  ./rfp_cache/<hash>.json
  (created automatically; one file per unique RFP PDF)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

_CACHE_DIR = Path("./rfp_cache")
_CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hash helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash_pdf(pdf_path: str) -> str:
    """SHA-256 of the first 512 KB (fast, unique enough for any real RFP)."""
    h = hashlib.sha256()
    try:
        with open(pdf_path, "rb") as f:
            h.update(f.read(512 * 1024))
    except OSError:
        return ""
    return h.hexdigest()[:16]


def _cache_path(pdf_hash: str) -> Path:
    return _CACHE_DIR / f"{pdf_hash}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(pdf_path: str) -> Optional[dict]:
    """
    Load cached extraction for this PDF.
    Returns None if no valid cache exists.
    """
    pdf_hash = _hash_pdf(pdf_path)
    if not pdf_hash:
        return None
    cp = _cache_path(pdf_hash)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        # Validate cache has essential fields
        if not data.get("criteria") or not data.get("grand_total"):
            print(f"[Cache] Cache file {cp.name} is incomplete — ignoring")
            return None
        age_s = (datetime.now() - datetime.fromisoformat(data.get("cached_at", "2000-01-01"))).total_seconds()
        print(f"[Cache] ✓ Loaded from cache: {cp.name} "
              f"({len(data['criteria'])} criteria, cached {int(age_s/3600)}h ago)")
        return data
    except Exception as e:
        print(f"[Cache] Error reading cache {cp}: {e}")
        return None


def save_cache(pdf_path: str, extraction: dict, bands: dict) -> bool:
    """
    Save extraction result + pre-computed bands to cache.
    extraction: the dict returned by extract_marking_table()
    bands: {parameter: [{"min": N, "max": M|null, "score": S}]}
    Returns True on success.
    """
    pdf_hash = _hash_pdf(pdf_path)
    if not pdf_hash:
        return False

    data = {
        "rfp_hash":     pdf_hash,
        "rfp_filename": Path(pdf_path).name,
        "cached_at":    datetime.now().isoformat(timespec="seconds"),
        "grand_total":  extraction.get("grand_total_marks", 0),
        "live_marks":   extraction.get("live_assessment_marks", 0),
        "live_label":   extraction.get("live_assessment_label", ""),
        "doc_total":    extraction.get("doc_max", 0),
        "threshold":    extraction.get("qualification_threshold_pct", 70.0),
        "criteria":     extraction.get("criteria", []),
        "bands":        bands,
        "context_source": extraction.get("context_source", ""),
    }

    cp = _cache_path(pdf_hash)
    try:
        cp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Cache] ✓ Saved to {cp.name} ({len(data['criteria'])} criteria, "
              f"{len(bands)} band sets)")
        return True
    except Exception as e:
        print(f"[Cache] Error writing cache {cp}: {e}")
        return False


def invalidate_cache(pdf_path: str) -> bool:
    """Delete cached extraction for this PDF (forces fresh extraction)."""
    pdf_hash = _hash_pdf(pdf_path)
    if not pdf_hash:
        return False
    cp = _cache_path(pdf_hash)
    if cp.exists():
        cp.unlink()
        print(f"[Cache] Invalidated: {cp.name}")
        return True
    return False


def get_bands(pdf_path: str) -> dict:
    """Return pre-computed bands dict, or {} if not cached."""
    data = load_cache(pdf_path)
    return data.get("bands", {}) if data else {}


# ─────────────────────────────────────────────────────────────────────────────
# Band pre-computation (called once, at cache-write time)
# ─────────────────────────────────────────────────────────────────────────────

_BAND_EXTRACTION_PROMPT = """\
Extract ONLY the scoring formula bands from this RFP criterion text.
Read the criteria_text carefully and convert every scoring threshold into a band.

CRITERION: {parameter}
FORMULA TYPE: {formula_type}
CRITERIA TEXT:
{criteria_text}

RULES:
- BAND: each "X to Y: N marks" → {{"min": X, "max": Y, "score": N}}
  - Upper bound is EXCLUSIVE (use the next band's min as max)
  - Open-ended top band (e.g. "more than 4: 10 marks") → max: null
  - If range is "100-200 Crs: 5 marks" → min:100, max:200, score:5
  - "300 Crs+: 15 marks" → min:300, max:null, score:15
- BINARY: present_score = max_marks, absent_score = 0
- STEP: base_threshold, base_score, step_size, step_score
- If no numeric bands exist, return empty bands list

Return ONLY valid JSON, no markdown:
{{
  "formula_type": "BAND|BINARY|STEP|QUAL|LLM",
  "bands": [{{"min": N, "max": M_or_null, "score": S}}],
  "present_score": null_or_number,
  "absent_score": 0,
  "notes": "any edge cases"
}}
"""


def precompute_bands(criteria: list[dict]) -> dict:
    """
    For every scoreable (non-parent) criterion, call the LLM ONCE to extract
    the scoring bands.  Returns {parameter: formula_dict}.

    This is called at cache-write time only — never at scoring time.
    """
    from core.llm_client import call_llm, extract_json

    bands: dict = {}
    scoreable = [c for c in criteria if not c.get("is_parent")]
    print(f"[Cache] Pre-computing bands for {len(scoreable)} criteria...")

    for c in scoreable:
        parameter    = c.get("parameter", "")
        formula_type = c.get("formula_type", "BAND")
        criteria_txt = c.get("criteria_text", "")[:600]

        if not criteria_txt:
            print(f"  [bands] No criteria_text for {parameter[:50]} — skipping")
            continue

        # BINARY: no need to call LLM
        if formula_type.upper() == "BINARY":
            max_marks = int(c.get("max_marks", 0))
            bands[parameter] = {
                "formula_type": "BINARY",
                "bands": [],
                "present_score": max_marks,
                "absent_score":  0,
            }
            print(f"  [bands] BINARY (no LLM): {parameter[:50]}")
            continue

        prompt = _BAND_EXTRACTION_PROMPT.format(
            parameter    = parameter,
            formula_type = formula_type,
            criteria_text= criteria_txt,
        )
        raw    = call_llm(prompt, label=f"bands-{parameter[:20]}")
        parsed = extract_json(raw) if raw else None

        if parsed and (parsed.get("bands") or parsed.get("present_score") is not None):
            bands[parameter] = parsed
            band_str = str(parsed.get("bands", []))[:80]
            print(f"  [bands] {parameter[:50]:50s} → {band_str}")
        else:
            # Fallback: try to parse bands directly from criteria_text using regex
            fallback = _regex_parse_bands(criteria_txt, formula_type)
            if fallback:
                bands[parameter] = fallback
                print(f"  [bands] {parameter[:50]:50s} → (regex fallback) {fallback['bands'][:2]}")
            else:
                print(f"  [bands] {parameter[:50]:50s} → FAILED (will use LLM at scoring time)")

    return bands


def _regex_parse_bands(criteria_text: str, formula_type: str) -> Optional[dict]:
    """
    Pure-regex band extraction as fallback when LLM fails.
    Handles patterns like:
      "100-200 Crs: 5 marks"
      "More than 4 projects: 10 marks"
      "500 to 1000 - 5 marks"
      "3 - 4 years of Experience: 4 marks"
    """
    # Pattern: "N to M: S marks" or "N-M: S marks"
    range_pat = re.compile(
        r'(\d[\d,]*(?:\.\d+)?)\s*(?:to|-|–)\s*(\d[\d,]*(?:\.\d+)?)\s*'
        r'(?:Crs?|crores?|years?|projects?|personnel|marks?)?\s*[:\-–]\s*'
        r'(\d+)\s*marks?',
        re.IGNORECASE,
    )
    # Pattern: "More than N: S marks" or "N+: S marks"
    open_pat = re.compile(
        r'(?:more\s+than|above|over|>\s*|greater\s+than)\s*(\d[\d,]*(?:\.\d+)?)'
        r'\s*(?:\w+\s*)?[:\-–]\s*(\d+)\s*marks?',
        re.IGNORECASE,
    )
    # Single threshold: "N+: S marks"
    plus_pat = re.compile(
        r'(\d[\d,]*(?:\.\d+)?)\s*\+\s*[:\-–]\s*(\d+)\s*marks?',
        re.IGNORECASE,
    )

    def clean_num(s: str) -> float:
        return float(s.replace(",", ""))

    bands: list[dict] = []

    for m in range_pat.finditer(criteria_text):
        lo = clean_num(m.group(1))
        hi = clean_num(m.group(2))
        sc = int(m.group(3))
        bands.append({"min": lo, "max": hi, "score": sc})

    for m in open_pat.finditer(criteria_text):
        lo = clean_num(m.group(1))
        sc = int(m.group(2))
        bands.append({"min": lo, "max": None, "score": sc})

    for m in plus_pat.finditer(criteria_text):
        lo = clean_num(m.group(1))
        sc = int(m.group(2))
        # Check if this is already covered
        if not any(abs(b["min"] - lo) < 0.1 and b["max"] is None for b in bands):
            bands.append({"min": lo, "max": None, "score": sc})

    if not bands:
        return None

    # Sort by min ascending
    bands.sort(key=lambda b: float(b["min"]) if b["min"] is not None else 0.0)

    return {
        "formula_type": formula_type.upper() if formula_type else "BAND",
        "bands":         bands,
        "present_score": None,
        "absent_score":  0,
        "notes":         "regex-extracted",
    }
