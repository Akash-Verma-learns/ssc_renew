"""
core/value_guard.py  —  Value Sanitiser
========================================

THE ROOT CAUSE THIS MODULE FIXES
──────────────────────────────────
The bulk extractor (`proposal_analyzer.bulk_extract_values`) searches the
proposal PDF for pages that contain scoring keywords.  Many proposals echo
the RFP's evaluation table — as a compliance matrix, a cover-page table, or
an appendix — so the TOP-SCORING pages for keywords like "experience" or
"turnover" are often the RFP criteria table itself, not the evidence section.

When this happens the LLM faithfully extracts the scoring-language value it
finds, e.g. "20 marks" for Past Experience B, and that string flows all the
way into the BAND scorer which parses it as the number 20, applies the
wrong band, and produces a corrupt score (5/20 instead of the actual score).

This bug is SILENT — the system reports a confident score with a page
reference and no warning, because "20" does appear on that page.

THIS MODULE provides a single function, `sanitise_extracted_value()`, that
is inserted between the bulk extractor and the scorer.  It:

  1. Rejects values that contain scoring/criteria language.
  2. Rejects values that are semantically impossible for the criterion type.
  3. Returns None (→ LLM fallback) rather than a corrupted value.

INTEGRATION  (one line each)
──────────────────────────────
In tq_scorer.py, score_criterion(), after the bulk extraction block:

    ev   = analysis.get_value(parameter) if analysis else None
    found = ...
    pg   = ...

    # ── ADD THIS ──────────────────────────────────────────────────────────
    from core.value_guard import sanitise_extracted_value
    ev, found = sanitise_extracted_value(ev, found, parameter, formula_hint)
    # ──────────────────────────────────────────────────────────────────────

    print(f"    [bulk] value={ev!r}  page={pg}  found={found}")
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Patterns that indicate the value came from scoring/criteria text, not from
# the bidder's actual claim.
# ─────────────────────────────────────────────────────────────────────────────

# "20 marks", "5 marks for 01 project", "maximum marks: 20"
_SCORING_LANGUAGE = re.compile(
    r'\b(marks?|points?|score[sd]?|criteria|maximum|minimum|max\.?|'
    r'out\s+of|evaluation|as\s+per|per\s+project|for\s+\d+\s+project)'
    r'\b',
    re.IGNORECASE,
)

# Sentinel strings returned when a value is genuinely absent
_NULL_SENTINELS = frozenset({
    "not found", "null", "none", "n/a", "na", "not stated",
    "not mentioned", "not provided", "not available", "not applicable",
    "not disclosed", "not given", "unknown", "unspecified", "–", "-",
})

# ─────────────────────────────────────────────────────────────────────────────
# Per-formula type sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_value(text: str) -> Optional[float]:
    """Extract the leading number from a string, or None."""
    m = re.search(r'\d[\d,\.]*', str(text).replace(",", ""))
    try:
        return float(m.group().replace(",", "")) if m else None
    except ValueError:
        return None


# Rough plausibility limits for common BAND criteria in Indian government RFPs.
# Values outside these ranges almost certainly came from a criteria table.
_BAND_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    # (min_plausible, max_plausible)
    "turnover":          (0.0,   50_000.0),   # Cr — 0 to 50,000 Cr
    "experience":        (0.0,      200.0),   # years
    "past experience a": (0.0,      500.0),   # number of professionals
    "past experience b": (0.0,      500.0),   # number of projects
    "project":           (0.0,      500.0),   # project count
    "revenue":           (0.0,   50_000.0),
    "net worth":         (0.0,   50_000.0),
}


def _plausibility_check(
    value_str: str,
    parameter: str,
    formula_type: str,
) -> bool:
    """
    Returns True if the numeric value is plausible for this criterion.
    Returns False if it is suspiciously small/large (likely a criteria score).
    """
    if formula_type.upper() not in ("BAND", "STEP"):
        return True  # only check numeric criteria

    num = _numeric_value(value_str)
    if num is None:
        return True  # can't check — let the scorer handle it

    param_lower = parameter.lower()

    # Match against known plausibility ranges
    for key, (lo, hi) in _BAND_PLAUSIBILITY.items():
        if key in param_lower:
            if not (lo <= num <= hi):
                return False
            break

    # Generic guard: values 1–100 that look like "marks" are suspicious
    # when the parameter contains "experience" or "project" but the value
    # string itself also mentions "marks" or "points".
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sanitise_extracted_value(
    value: Optional[str],
    found: bool,
    parameter: str,
    formula_type: str,
) -> Tuple[Optional[str], bool]:
    """
    Validate an extracted value before it reaches the scorer.

    Returns (sanitised_value, sanitised_found).
    Returns (None, False) to signal "treat as not found → use LLM fallback".

    Rules applied in order:
      1. None / empty / null sentinel  → not found
      2. Contains scoring language     → reject (criteria table echo)
      3. Numeric plausibility check    → reject if out of range
    """
    if not value:
        return None, False

    v = value.strip()

    # Rule 1 — null sentinel
    if v.lower() in _NULL_SENTINELS:
        return None, False

    # Rule 2 — scoring language present in value
    if _SCORING_LANGUAGE.search(v):
        print(
            f"    [value_guard] Rejected '{v[:60]}' for '{parameter[:40]}'"
            f" — looks like scoring/criteria text, not a bidder claim."
        )
        return None, False

    # Rule 3 — numeric plausibility (BAND/STEP only)
    if not _plausibility_check(v, parameter, formula_type):
        print(
            f"    [value_guard] Rejected '{v[:60]}' for '{parameter[:40]}'"
            f" — numeric value implausible for {formula_type} criterion."
        )
        return None, False

    return v, found
