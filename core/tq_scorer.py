"""
core/tq_scorer.py  —  v6  Bulk-First Architecture (Local-Model Optimised)
==========================================================================

THE ONCE-AND-FOR-ALL REWRITE
─────────────────────────────
The old scorer called the LLM once per criterion for extraction,
then again for formula, then again for scoring fallback.
With 14 criteria = 28-42 sequential local LLM calls.
Local models are slow and inconsistent at this frequency.

v6 ARCHITECTURE:
────────────────
  BEFORE SCORING:
    proposal_analyzer.analyze_proposal() runs ONCE:
      - 1 bulk LLM call → values for all BAND/BINARY/STEP criteria
      - 0 LLM calls     → pure-Python CV detection for all QUAL criteria
      - Results cached  → subsequent runs: 0 LLM calls at all

  PER CRITERION:
    score_criterion() reads from the pre-analyzed ProposalAnalysis.
    No LLM calls during scoring in the happy path.
    LLM fallback only fires if value not found AND formula not cached.

  QUAL CRITERIA (Team Leader, GIS Expert, etc.):
    BINARY: CV present in proposal → full marks. Not found → 0.
    Confidence levels affect score:
      high   → 100% of max_marks
      medium →  75% of max_marks
      low    →  25% of max_marks (but evidence found)
      absent →  0

  BAND/BINARY/STEP CRITERIA:
    Value from ProposalAnalysis → apply cached bands → deterministic score.

PUBLIC API (unchanged)
──────────────────────
  score_criterion(criterion, proposal_path, all_criteria=None) -> dict
  warm_analysis_cache(proposal_path, criteria) -> ProposalAnalysis
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from core.llm_client import call_llm, extract_json

try:
    from core.proposal_analyzer import analyze_proposal, ProposalAnalysis
    _ANALYZER_OK = True
except ImportError:
    _ANALYZER_OK = False
    print("[Scorer-v6] WARNING: proposal_analyzer not found")

# ─────────────────────────────────────────────────────────────────────────────
# Module-level analysis cache (per proposal, populated before scoring)
# ─────────────────────────────────────────────────────────────────────────────

_analysis_cache: dict[str, "ProposalAnalysis"] = {}


def warm_analysis_cache(proposal_path: str, criteria: list[dict]) -> Optional["ProposalAnalysis"]:
    """
    Pre-populate the analysis cache before scoring begins.
    Call this from the orchestrator once, before the scoring loop.
    Returns the ProposalAnalysis so the caller can log it.
    """
    if not _ANALYZER_OK:
        return None
    if proposal_path not in _analysis_cache:
        _analysis_cache[proposal_path] = analyze_proposal(proposal_path, criteria)
    return _analysis_cache[proposal_path]


def _get_analysis(proposal_path: str, criteria: list[dict]) -> Optional["ProposalAnalysis"]:
    if not _ANALYZER_OK:
        return None
    if proposal_path not in _analysis_cache:
        _analysis_cache[proposal_path] = analyze_proposal(proposal_path, criteria)
    return _analysis_cache[proposal_path]


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic formula application
# ─────────────────────────────────────────────────────────────────────────────

def _parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    clean = str(text).replace(",", "")
    m = re.search(r'\d+(?:\.\d+)?', clean)
    return float(m.group()) if m else None


def _has_more_than(text: str) -> bool:
    return bool(re.search(
        r'\b(more\s+than|at\s+least|over\s+|greater\s+than|above\b|minimum|min\.?\s|\d+\+)',
        str(text), re.I,
    ))


def _apply_band(bands: list, value: float, max_marks: int,
                value_str: str = "") -> float:
    """
    Correct boundary: [lo, hi) with open-ended upper band.
    "More than N" → effective value = N + 1.
    Walk all bands ascending; last match wins.
    """
    if not bands:
        return 0.0

    effective = value + 1.0 if (value_str and _has_more_than(value_str)) else value
    matched: Optional[float] = None

    for band in sorted(bands, key=lambda b: float(b.get("min") or 0)):
        lo    = float(band.get("min") if band.get("min") is not None else float("-inf"))
        hi_raw = band.get("max")
        hi    = float(hi_raw) if hi_raw is not None else float("inf")
        sc    = float(band.get("score", 0))

        if hi_raw is None:
            if effective >= lo:
                matched = sc
        else:
            if lo <= effective < hi:
                matched = sc

    return round(min(matched or 0.0, float(max_marks)), 1)


def _apply_binary(found: bool, value_str: str, max_marks: int,
                  present_score: Optional[float] = None) -> float:
    absent_signals = {"not found", "no", "absent", "none", "not present",
                      "not mentioned", "not stated", "null", ""}
    v_low = (value_str or "").lower().strip()
    is_present = found and v_low not in absent_signals
    score = float(present_score if present_score is not None else max_marks)
    return round(min(score, float(max_marks)), 1) if is_present else 0.0


def _deterministic_score(
    bands_formula: Optional[dict],
    value_str: Optional[str],
    found: bool,
    max_marks: int,
    formula_hint: str,
) -> Optional[float]:
    if not bands_formula:
        return None

    ftype = (bands_formula.get("formula_type") or formula_hint or "LLM").upper()
    num   = _parse_number(value_str or "")

    if ftype == "BAND" and num is not None:
        bands = bands_formula.get("bands") or []
        if bands:
            return _apply_band(bands, num, max_marks, value_str or "")

    elif ftype == "BINARY":
        return _apply_binary(found, value_str or "", max_marks,
                             bands_formula.get("present_score"))

    elif ftype == "STEP":
        try:
            bt = float(bands_formula["base_threshold"])
            bs = float(bands_formula["base_score"])
            ss = float(bands_formula["step_size"])
            sv = float(bands_formula["step_score"])
            if ss > 0 and num is not None and num >= bt:
                steps = int((num - bt) / ss)
                return round(min(bs + steps * sv, float(max_marks)), 1)
            return 0.0 if num is not None and num < bt else None
        except (KeyError, ValueError, TypeError):
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Verification (lightweight)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_on_page(proposal_path: str, value_str: str, page: Optional[int]) -> bool:
    if not page or not value_str:
        return True
    try:
        import fitz
        doc     = fitz.open(proposal_path)
        if page < 1 or page > len(doc):
            doc.close()
            return True
        pg_text = doc[page - 1].get_text().lower()
        doc.close()
        nums = [n.replace(",", "") for n in re.findall(r'\d[\d,.]*', value_str)]
        if nums:
            return any(n in pg_text for n in nums[:3])
        words = re.findall(r'[a-z]{4,}', value_str.lower())
        return any(w in pg_text for w in words[:3]) if words else True
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# QUAL scoring (pure-Python, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

_CONFIDENCE_MULTIPLIER = {"high": 1.0, "medium": 0.75, "low": 0.25}


def _score_qual_from_analysis(
    analysis: Optional["ProposalAnalysis"],
    criterion: dict,
    proposal_path: str,
) -> dict:
    """
    Score QUAL/expert-role criterion using pure-Python CV detection.
    No LLM call needed.
    """
    parameter = criterion.get("parameter", "")
    max_marks = int(criterion.get("max_marks") or 0)

    # Get CV detection result
    if analysis:
        cv = analysis.get_cv(parameter)
    else:
        # Direct detection (no cache)
        try:
            from core.proposal_analyzer import detect_cv_for_role
            cv = detect_cv_for_role(proposal_path, parameter)
        except Exception:
            cv = {"present": False, "confidence": "low",
                  "evidence_page": None, "evidence_snippet": ""}

    present    = cv.get("present", False)
    confidence = cv.get("confidence", "low")
    ev_page    = cv.get("evidence_page")
    snippet    = cv.get("evidence_snippet", "")

    if not present:
        return {
            "score":           0.0,
            "extracted_value": "Not found in proposal",
            "source_page":     None,
            "scoring_steps":   f"CV detection: role not found (0/{max_marks})",
            "justification":   f"No CV evidence found for: {parameter}",
            "strengths":       [],
            "gaps":            [f"CV/profile for {parameter} not found in proposal"],
            "evidence_found":  False,
            "verified":        True,
            "source":          "cv_detection",
        }

    multiplier = _CONFIDENCE_MULTIPLIER.get(confidence, 0.25)
    score      = round(max_marks * multiplier, 1)
    # Round to nearest 0.5 for cleaner output
    score      = round(score * 2) / 2

    return {
        "score":           score,
        "extracted_value": f"CV found (confidence: {confidence})",
        "source_page":     ev_page,
        "scoring_steps":   f"CV detection [{confidence}]: {score}/{max_marks}",
        "justification":   f"CV/profile evidence found for {parameter} with {confidence} confidence (p.{ev_page})",
        "strengths":       [f"Expert profile found (p.{ev_page}): {snippet[:80]}"] if snippet else
                           [f"Expert profile found (p.{ev_page})"],
        "gaps":            [] if confidence == "high" else
                           [f"CV confidence {confidence} — verify profile completeness"],
        "evidence_found":  True,
        "verified":        True,
        "source":          "cv_detection",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM fallback (only fires when bulk extraction missed a criterion)
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_PROMPT = """\
Extract the EXACT value claimed in this proposal for ONE criterion.

CRITERION: {parameter}
MAX MARKS: {max_marks}
SCORING RULE: {criteria_text}

PROPOSAL PAGES:
{pages}

Return ONLY valid JSON:
{{"found": true/false, "value": "<exact value or null>", "page": <page number or null>}}
"""


def _llm_fallback_extract(criterion: dict, proposal_path: str) -> dict:
    """Single-criterion LLM extraction — only called when bulk missed this one."""
    try:
        from core.proposal_analyzer import _get_top_pages_for_criteria
        pages_text, _ = _get_top_pages_for_criteria(
            proposal_path, [criterion], max_total_chars=3000)
    except Exception:
        pages_text = ""

    if not pages_text:
        return {"found": False, "value": None, "page": None}

    prompt = _FALLBACK_PROMPT.format(
        parameter    = criterion.get("parameter", ""),
        max_marks    = criterion.get("max_marks", 0),
        criteria_text= criterion.get("criteria_text", "")[:300],
        pages        = pages_text[:2500],
    )
    raw    = call_llm(prompt, label=f"fallback-{criterion.get('parameter','?')[:20]}")
    parsed = extract_json(raw) if raw else None
    if not parsed:
        return {"found": False, "value": None, "page": None}
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zero(reason: str) -> dict:
    return {
        "score": 0.0, "extracted_value": None, "source_page": None,
        "scoring_steps": reason, "justification": reason,
        "strengths": [], "gaps": [reason], "evidence_found": False,
        "verified": True, "source": "none",
    }


def _make_result(score: float, ev: str, pg: Optional[int],
                 steps: str, max_marks: int,
                 verified: bool = True, source: str = "bulk") -> dict:
    final = round(max(0.0, min(score, float(max_marks))), 1)
    return {
        "score":           final,
        "extracted_value": ev,
        "source_page":     pg,
        "scoring_steps":   steps,
        "justification":   f"Score {final}/{max_marks}. Found: {ev}"
                           + (f" (p.{pg})" if pg else "")
                           + ("" if verified else " ⚠ unverified"),
        "strengths":       [f"Found: {ev}" + (f" (p.{pg})" if pg else "")] if final > 0 else [],
        "gaps":            ([] if final >= max_marks
                            else ["Additional evidence needed for full marks"]),
        "evidence_found":  final > 0,
        "verified":        verified,
        "source":          source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
from core.value_guard import sanitise_extracted_value
def score_criterion(
    criterion: dict,
    proposal_path: str,
    all_criteria: Optional[list] = None,
) -> dict:
    """
    Score one criterion against the proposal PDF.

    v6 Pipeline:
      1. QUAL/LLM criteria → pure-Python CV detection (0 LLM calls)
      2. BAND/BINARY/STEP  → read from ProposalAnalysis (pre-computed)
         → apply cached RFP bands (pure Python)
      3. If value missing  → 1 LLM fallback call (per-criterion, last resort)
    """
    max_marks    = int(criterion.get("max_marks") or 0)
    parameter    = criterion.get("parameter", "")
    formula_hint = (criterion.get("formula_type") or "LLM").upper()
    is_parent    = criterion.get("is_parent", False)
    cached_bands: Optional[dict] = criterion.get("_cached_bands")

    if max_marks == 0:
        return _zero("Zero-mark criterion")
    if is_parent:
        return _zero("Parent criterion — scored via sub-criteria")
    if not Path(proposal_path).exists():
        return _zero(f"Proposal file not found: {proposal_path}")

    # Get the analysis (either from module cache or compute it)
    analysis = _get_analysis(proposal_path, all_criteria or [criterion])

    # ── QUAL / LLM: CV detection path (no LLM) ────────────────────────────
    if formula_hint in ("QUAL", "LLM"):
        return _score_qual_from_analysis(analysis, criterion, proposal_path)

    # ── BAND / BINARY / STEP: bulk extraction path ─────────────────────────
    # ev   = analysis.get_value(parameter) if analysis else None
    # found = (analysis.get_found(parameter) if analysis else False) or bool(ev)
    # pg   = analysis.get_page(parameter) if analysis else None

    # print(f"    [bulk] value={ev!r}  page={pg}  found={found}")
    ev    = analysis.get_value(parameter) if analysis else None
    found = (analysis.get_found(parameter) if analysis else False) or bool(ev)
    pg    = analysis.get_page(parameter) if analysis else None

   # Guard: reject values that look like scoring/criteria text (e.g. "20 marks")
    ev, found = sanitise_extracted_value(ev, found, parameter, formula_hint)

    print(f"    [bulk] value={ev!r}  page={pg}  found={found}")

    if not found or not ev:
        # Fallback: one targeted LLM call
        print(f"    [fallback] Bulk missed '{parameter}' — calling LLM")
        fb = _llm_fallback_extract(criterion, proposal_path)
        ev    = fb.get("value") or "Not found"
        found = bool(fb.get("found"))
        pg    = fb.get("page") or pg

    if not found or not ev or ev.lower() in ("not found", "null", "none", ""):
        return _zero(f"Evidence not found for: {parameter}")

    # Try deterministic scoring with cached bands
    score: Optional[float] = None
    bands_used = "none"

    if cached_bands and (cached_bands.get("bands") or
                         cached_bands.get("formula_type","").upper() == "BINARY"):
        score = _deterministic_score(cached_bands, ev, found, max_marks, formula_hint)
        bands_used = "cached"

    if score is None:
        # Try to get bands from RFP cache
        try:
            from core.rfp_cache import get_bands
            rfp_bands = get_bands(proposal_path)  # usually empty — needs RFP path
        except Exception:
            rfp_bands = {}

        # Last resort: construct minimal formula from formula_hint
        if formula_hint == "BINARY":
            score = _apply_binary(found, ev, max_marks)
            bands_used = "formula_hint"
        elif formula_hint in ("BAND", "STEP") and _parse_number(ev) is not None:
            # Can't score without bands — use LLM direct score (single call)
            num = _parse_number(ev)
            print(f"    [no-bands] No cached bands for BAND criterion — LLM score")
            score_prompt = (
                f"Score this value against the RFP criterion.\n"
                f"CRITERION: {parameter}\n"
                f"MAX MARKS: {max_marks}\n"
                f"SCORING RULES: {criterion.get('criteria_text','')[:300]}\n"
                f"BIDDER VALUE: {ev}\n"
                f"Return ONLY valid JSON: {{\"score\": <0-{max_marks}>}}"
            )
            raw2   = call_llm(score_prompt, label=f"band-score-{parameter[:15]}")
            parsed = extract_json(raw2) if raw2 else {}
            score  = float(parsed.get("score", 0)) if parsed else 0.0
            score  = round(max(0.0, min(score, float(max_marks))), 1)
            bands_used = "llm_direct"

    if score is None:
        return _zero(f"Could not score: {parameter} (value={ev})")

    # Verify on source page
    verified = _verify_on_page(proposal_path, ev, pg)
    steps    = f"{formula_hint} ({bands_used}): value={ev!r} → {score}/{max_marks}"
    print(f"    [score] {steps}" + ("" if verified else " ⚠ unverified"))

    return _make_result(score, ev, pg, steps, max_marks,
                        verified=verified, source=f"bulk_{bands_used}")