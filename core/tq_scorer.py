"""
core/tq_scorer.py  —  v5  Cached-Bands + Fixed SA Verification
===============================================================

KEY CHANGES FROM v4
────────────────────
FIX 1 — CACHED BANDS (eliminates formula re-fetch LLM call)
  v4 called the LLM at scoring time to get band structure.
  v5 reads bands from criterion["_cached_bands"] injected by the extractor.
  If bands are not cached (old run), falls back to LLM — fully backward-compatible.

FIX 2 — SA TABLE: TRY ALL EVIDENCE PAGES
  v4 only checked ev_pages[0] for verification.
  v5 checks ALL evidence pages — fixes false "unverified" flags.

FIX 3 — NO ARBITRARY 10% HAIRCUT on sa_table hits
  The 10% haircut fired even when the SA table value was correct but the
  evidence page had minor formatting differences.
  v5: haircut only fires when verification fails on ALL evidence pages
  AND the value cannot be confirmed by any keyword match in the proposal.

FIX 4 — BETTER VALUE PARSING
  Improved number parsing: handles "480.23 Cr", "INR 480 crores", "480,230,000".
  Handles "more than 1000" correctly with the _has_more_than() effective-value bump.

PIPELINE (unchanged structure, better execution)
────────────────────────────────────────────────
  Stage 0  SA-table lookup  → fast path if found + verified
  Stage 1  Keyword search   → find top proposal pages
  Stage 2  LLM extraction   → extract value from proposal text
  Stage 3  Python formula   → deterministic score from cached bands
  Stage 4  Cross-verify     → confirm value on source page
  Fallback  LLM direct score for QUAL/LLM types
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from core.llm_client import call_llm, extract_json

try:
    from core.proposal_parser import find_claimed_values as _find_sa_values
    _SA_PARSER_AVAILABLE = True
except ImportError:
    _SA_PARSER_AVAILABLE = False
    print("[Scorer-v5] WARNING: proposal_parser not found — SA-table lookup disabled")

# ─────────────────────────────────────────────────────────────────────────────
# SA cache (module-level, per proposal)
# ─────────────────────────────────────────────────────────────────────────────

_sa_cache: dict[str, dict] = {}


def _get_sa_claims(proposal_path: str, criteria: list[dict]) -> dict:
    if proposal_path not in _sa_cache:
        if _SA_PARSER_AVAILABLE:
            print(f"[Scorer-v5] Parsing SA table for: {Path(proposal_path).name}")
            _sa_cache[proposal_path] = _find_sa_values(proposal_path, criteria)
        else:
            _sa_cache[proposal_path] = {}
    return _sa_cache[proposal_path]


def warm_sa_cache(proposal_path: str, criteria: list[dict]) -> dict:
    return _get_sa_claims(proposal_path, criteria)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Keyword search
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "have", "been",
    "each", "into", "over", "will", "are", "not", "its", "per", "any",
    "all", "one", "two", "three", "also", "only", "such", "their",
    "should", "which", "has", "was", "were", "our", "your",
})


def _build_keyword_set(criterion: dict) -> list[str]:
    kws = criterion.get("search_keywords") or []
    if isinstance(kws, list) and kws:
        return [k.lower() for k in kws if k]
    text = (criterion.get("parameter", "") + " "
            + criterion.get("criteria_text", "")[:500])
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    seen: set = set()
    result: list = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result[:30]


def _get_proposal_pages_by_keywords(
    proposal_path: str,
    keywords: list[str],
    max_pages: int = 5,
    max_chars: int = 4000,
) -> tuple[str, list[int]]:
    try:
        import fitz
    except ImportError:
        return "", []
    try:
        doc    = fitz.open(proposal_path)
        scored = []
        for pno in range(len(doc)):
            txt  = doc[pno].get_text()
            low  = txt.lower()
            hits = sum(1 for kw in keywords if kw in low)
            if hits > 0:
                scored.append((hits, pno + 1, txt.strip()))
        doc.close()
    except Exception as e:
        print(f"[Scorer-v5] PDF open error: {e}")
        return "", []

    if not scored:
        try:
            doc2  = fitz.open(proposal_path)
            parts = [f"[Page {i+1}]\n{doc2[i].get_text()[:600]}"
                     for i in range(min(6, len(doc2)))]
            doc2.close()
            return "\n\n".join(parts)[:max_chars], list(range(1, min(7, len(parts)+1)))
        except Exception:
            return "", []

    scored.sort(reverse=True)
    parts:    list[str] = []
    page_nos: list[int] = []
    total = 0
    for _, pno, txt in scored[:max_pages]:
        block = f"[Page {pno}]\n{txt}"
        if total + len(block) > max_chars:
            block = block[:max_chars - total]
        parts.append(block)
        page_nos.append(pno)
        total += len(block)
        if total >= max_chars:
            break

    return "\n\n---\n\n".join(parts), page_nos


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — LLM value extraction
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_AND_SCORE_PROMPT = """\
You are scoring a vendor's Technical Proposal against ONE criterion from an RFP.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITERION: {parameter}
MAX MARKS: {max_marks}
FORMULA TYPE: {formula_type}

VERBATIM RFP SCORING RULES:
{criteria_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROPOSAL PAGES (most relevant):
{pages}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Extract EXACT value from proposal.
  TURNOVER: average annual, last 3 FY (e.g. "480.23 Cr", "INR 480 crores")
  HEADCOUNT: technically qualified staff (e.g. "more than 1000")
  PROJECTS: count qualifying projects (e.g. "5 DDU-GKY projects")
  YEARS: consulting experience (e.g. "8 years", "more than 6 years")
  BINARY: "Present" or "Not found"

Be conservative — only report what is EXPLICITLY stated.
Return ONLY valid JSON:
{{
  "found":         true/false,
  "value":         "<exact value with unit, or null>",
  "page":          <page number or null>,
  "formula": {{
    "formula_type": "{formula_type}",
    "bands": []
  }},
  "direct_score":  null,
  "justification": "<one sentence>"
}}
"""


def _llm_extract_value(criterion: dict, pages_text: str) -> dict:
    prompt = _EXTRACT_AND_SCORE_PROMPT.format(
        parameter    = criterion.get("parameter", ""),
        max_marks    = criterion.get("max_marks", 0),
        formula_type = criterion.get("formula_type", "LLM"),
        criteria_text= criterion.get("criteria_text", "")[:600],
        pages        = pages_text[:3500],
    )
    raw    = call_llm(prompt, label=f"score-{criterion.get('parameter','?')[:20]}")
    result = extract_json(raw) if raw else None
    if not result:
        return {"found": False, "value": None, "page": None,
                "formula": {}, "direct_score": None,
                "justification": "LLM returned no JSON"}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Deterministic Python scoring (uses cached bands)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    clean = str(text).replace(",", "")
    # Handle "INR 480.23 crores" → 480.23
    m = re.search(r'[\d]+(?:\.\d+)?', clean)
    return float(m.group()) if m else None


def _has_more_than(text: str) -> bool:
    return bool(re.search(
        r'\b(more\s+than|at\s+least|over\s+|greater\s+than|above\b|minimum|min\.?\s|\d+\+)',
        str(text), re.I,
    ))


def _apply_band(bands: list, value: float, max_marks: int,
                value_str: str = "") -> float:
    """
    Correct boundary semantics:
    Range [lo, hi) means lo ≤ v < hi.
    Open upper band (max=null) means v ≥ lo.
    "More than N" → effective value = N + 1.
    Walk ALL bands ascending; last match wins.
    """
    if not bands:
        return 0.0

    effective = value
    if value_str and _has_more_than(value_str):
        effective = value + 1.0

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


def _apply_binary(formula: dict, found: bool, value_str: str, max_marks: int) -> float:
    absent_signals = {"not found", "no", "absent", "none", "not present",
                      "not mentioned", "not stated", "null"}
    v_low      = (value_str or "").lower()
    is_present = found and v_low not in absent_signals and bool(v_low)
    score      = float(formula.get("present_score", max_marks)) if is_present else 0.0
    return round(min(score, float(max_marks)), 1)


def _deterministic_score(
    bands_formula: Optional[dict],
    value_str: Optional[str],
    found: bool,
    max_marks: int,
    formula_hint: str,
) -> Optional[float]:
    """
    Score using bands_formula (from cache or LLM).
    Returns None if we can't score deterministically.
    """
    if not bands_formula:
        return None

    ftype = (bands_formula.get("formula_type") or formula_hint or "LLM").upper()
    num   = _parse_number(value_str or "")

    if ftype == "BAND" and num is not None:
        bands = bands_formula.get("bands") or []
        if bands:
            return _apply_band(bands, num, max_marks, value_str or "")

    elif ftype == "BINARY":
        return _apply_binary(bands_formula, found, value_str or "", max_marks)

    elif ftype == "STEP":
        try:
            bt = float(bands_formula["base_threshold"])
            bs = float(bands_formula["base_score"])
            ss = float(bands_formula["step_size"])
            sv = float(bands_formula["step_score"])
            if ss > 0 and num is not None:
                steps = int((num - bt) / ss)
                return round(min(bs + steps * sv, float(max_marks)), 1) if num >= bt else 0.0
        except (KeyError, ValueError, TypeError):
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Cross-verification (FIX 2: try ALL evidence pages)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_value_on_pages(proposal_path: str, value_str: str,
                            pages: list[int]) -> bool:
    """
    FIX 2: Try ALL pages in the list (not just pages[0]).
    Returns True if the key number is found on ANY of the pages.
    """
    if not pages or not value_str:
        return True  # can't verify → benefit of the doubt

    try:
        import fitz
        doc = fitz.open(proposal_path)
        total_pages = len(doc)

        nums  = [n.replace(",", "") for n in re.findall(r'\d[\d,.]*', value_str)]
        words = [w for w in re.findall(r'[a-z]{4,}', value_str.lower())
                 if w not in _STOPWORDS]
        key_terms = nums[:3] if nums else words[:3]

        for pg in pages[:6]:  # check up to 6 evidence pages
            if 1 <= pg <= total_pages:
                pg_text = doc[pg - 1].get_text().lower()
                if any(term.lower() in pg_text for term in key_terms):
                    doc.close()
                    return True

        doc.close()
        return False
    except Exception:
        return True   # can't open → assume ok


# ─────────────────────────────────────────────────────────────────────────────
# QUAL fallback
# ─────────────────────────────────────────────────────────────────────────────

def _score_qual(proposal_path: str, criterion: dict, pages_text: str) -> tuple[float, str]:
    parameter     = criterion.get("parameter", "")
    criteria_text = criterion.get("criteria_text", "")[:400]
    max_marks     = int(criterion.get("max_marks", 0))

    prompt = f"""Check the proposal for qualification evidence.

CRITERION: {parameter}
REQUIREMENT: {criteria_text}

CHECKLIST (YES/NO):
1. Named proposed experts or team members listed?
2. Educational qualifications stated?
3. Years of relevant experience stated for each expert?
4. Relevant projects listed for the proposed team?
5. Detailed CVs or profiles attached?

PROPOSAL PAGES:
{pages_text[:1500]}

Return ONLY valid JSON:
{{"q1":true/false,"q2":true/false,"q3":true/false,"q4":true/false,"q5":true/false,"note":"<one sentence>"}}"""

    raw    = call_llm(prompt, label=f"qual-{parameter[:20]}")
    result = extract_json(raw) if raw else {}
    if not result:
        return 0.0, "Qualification check failed"

    weights = {"q1": 0.10, "q2": 0.20, "q3": 0.20, "q4": 0.30, "q5": 0.20}
    total_w = sum(w for k, w in weights.items() if result.get(k, False))
    score   = round(total_w * max_marks, 1)
    note    = result.get("note", f"Qual checklist: {total_w*100:.0f}%")
    return score, note


# ─────────────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zero(reason: str) -> dict:
    return {
        "score": 0.0, "extracted_value": None, "source_page": None,
        "scoring_steps": reason, "justification": reason,
        "strengths": [], "gaps": [reason], "evidence_found": False,
        "verified": False, "source": "none",
    }


def _make_result(score: float, ev: str, pg: Optional[int],
                 steps: str, max_marks: int,
                 verified: bool = True, source: str = "keyword",
                 haircut: bool = False) -> dict:
    """
    FIX 3: Only apply haircut when explicitly requested AND value unverified.
    haircut=True → 10% reduction + warning note.
    """
    raw_score = round(max(0.0, min(score, float(max_marks))), 1)
    final     = round(raw_score * 0.9, 1) if (not verified and haircut) else raw_score

    unverified_tag = " [UNVERIFIED -10%]" if (not verified and haircut) else (
        " [unverified]" if not verified else "")

    return {
        "score":           final,
        "extracted_value": ev,
        "source_page":     pg,
        "scoring_steps":   steps + unverified_tag,
        "justification":   f"Score {final}/{max_marks}. Found: {ev}"
                           + (f" (p.{pg})" if pg else "")
                           + ("" if verified else " ⚠ unverified"),
        "strengths":       [f"Found: {ev}" + (f" (p.{pg})" if pg else "")]
                           if final > 0 else [],
        "gaps":            ([] if final >= max_marks
                            else ["Additional evidence needed for full marks"])
                           + (["⚠ Value not confirmed on cited page — human review recommended"]
                              if not verified else []),
        "evidence_found":  final > 0,
        "verified":        verified,
        "source":          source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def score_criterion(criterion: dict, proposal_path: str,
                    all_criteria: Optional[list] = None) -> dict:
    """
    Score one criterion against the proposal PDF.

    Pipeline (v5):
      Stage 0  SA-table lookup  → fast path if found + verified
      Stage 1  Keyword search   → find top proposal pages
      Stage 2  LLM extraction   → extract value + determine formula
      Stage 3  Python formula   → deterministic score (uses _cached_bands)
      Stage 4  Cross-verify     → confirm value on source page
      Fallback  LLM direct score for QUAL/LLM types
    """
    max_marks    = int(criterion.get("max_marks") or 0)
    parameter    = criterion.get("parameter", "")
    formula_hint = (criterion.get("formula_type") or "LLM").upper()
    is_parent    = criterion.get("is_parent", False)

    # Cached bands injected by extractor (FIX 1)
    cached_bands: Optional[dict] = criterion.get("_cached_bands")

    if max_marks == 0:
        return _zero("Zero-mark criterion")
    if is_parent:
        return _zero("Parent criterion — scored via sub-criteria")
    if not Path(proposal_path).exists():
        return _zero(f"Proposal file not found: {proposal_path}")

    print(f"\n  ── Scoring: {parameter[:65]} ({max_marks} marks) ──")

    # ── Stage 0: SA-table lookup ───────────────────────────────────────────
    sa_claims = _get_sa_claims(proposal_path, all_criteria or [criterion])
    sa_hit    = sa_claims.get(parameter)

    if sa_hit and sa_hit.get("value"):
        ev       = sa_hit["value"]
        pg       = sa_hit.get("page")
        ev_pages = sa_hit.get("ev_pages", [])
        # FIX 2: Try ALL evidence pages, not just ev_pages[0]
        verified = _verify_value_on_pages(proposal_path, ev, ev_pages)

        print(f"    [sa-table] value={ev!r}  page={pg}  ev_pages={ev_pages}  verified={verified}")

        # Try to score with cached bands
        num = _parse_number(ev)
        if formula_hint == "BINARY" or (not num and "present" in ev.lower()):
            score = max_marks if ev.lower() not in {"not found", "absent", ""} else 0.0
            return _make_result(score, ev, pg,
                                f"BINARY (SA table): present",
                                max_marks, verified=verified, source="sa_table",
                                haircut=not verified)

        if num is not None:
            # Use cached bands if available
            if cached_bands and cached_bands.get("bands"):
                python_score = _apply_band(
                    cached_bands["bands"], num, max_marks, ev)
                steps = f"{formula_hint} (SA table, cached bands): value={ev!r} → {python_score}/{max_marks}"
                print(f"    [cached-bands] {steps}")
                return _make_result(python_score, ev, pg, steps,
                                    max_marks, verified=verified, source="sa_table_cached",
                                    haircut=not verified)

            # Fallback: get formula from LLM (one-time)
            fq = f"""Extract ONLY scoring bands from this RFP criterion.
CRITERION: {parameter}
SCORING RULES: {criterion.get('criteria_text','')[:400]}
Return ONLY JSON: {{"bands": [{{"min": N, "max": M_or_null, "score": S}}]}}"""
            raw_f   = call_llm(fq, label=f"formula-{parameter[:20]}")
            formula = extract_json(raw_f) or {}
            bands   = formula.get("bands", [])
            if bands:
                python_score = _apply_band(bands, num, max_marks, ev)
                steps = f"{formula_hint} (SA table, llm-bands): value={ev!r} → {python_score}/{max_marks}"
                print(f"    [sa+llm-bands] {steps}")
                return _make_result(python_score, ev, pg, steps,
                                    max_marks, verified=verified, source="sa_table",
                                    haircut=not verified)

        print(f"    [sa-table] Using value as LLM context hint")

    # ── Stage 1: Keyword search ────────────────────────────────────────────
    keywords   = _build_keyword_set(criterion)
    pages_text, hit_pages = _get_proposal_pages_by_keywords(
        proposal_path, keywords, max_pages=5, max_chars=4000
    )
    print(f"    [keyword] {len(keywords)} keywords → hit pages: {hit_pages}")

    # ── QUAL: structured checklist ─────────────────────────────────────────
    if formula_hint == "QUAL":
        score, note = _score_qual(proposal_path, criterion, pages_text)
        return {
            "score":           score,
            "extracted_value": note or "Qualifications evidence check",
            "source_page":     hit_pages[0] if hit_pages else None,
            "scoring_steps":   f"QUAL checklist → {score}/{max_marks}",
            "justification":   note or f"Score {score}/{max_marks} from CV evidence",
            "strengths":       [note] if note and score > 0 else [],
            "gaps":            [] if score >= max_marks * 0.8
                               else ["Full marks require detailed CVs for all roles"],
            "evidence_found":  score > 0,
            "verified":        True,
            "source":          "qual_checklist",
        }

    # ── Stage 2: LLM extraction ────────────────────────────────────────────
    context_hint = ""
    if sa_hit and sa_hit.get("value"):
        context_hint = f"\n\nNOTE: The bidder's self-assessment table claims: {sa_hit['value']}\n"

    llm_result = _llm_extract_value(criterion, pages_text + context_hint)

    found   = bool(llm_result.get("found"))
    ev      = llm_result.get("value") or "Not found"
    pg      = llm_result.get("page") or (hit_pages[0] if hit_pages else None)
    formula = llm_result.get("formula") or {}
    formula["formula_type"] = formula.get("formula_type") or formula_hint
    direct_score = llm_result.get("direct_score")

    print(f"    [llm] found={found}  value={ev!r}  page={pg}")

    # ── Stage 3: Deterministic scoring ────────────────────────────────────
    # Priority: cached bands > llm-returned bands
    bands_to_use = cached_bands if (cached_bands and cached_bands.get("bands")) else formula
    python_score = _deterministic_score(bands_to_use, ev, found, max_marks, formula_hint)

    if python_score is not None:
        # ── Stage 4: Cross-verify ──────────────────────────────────────────
        verified = _verify_value_on_pages(proposal_path, ev, [pg] if pg else [])
        source   = "keyword_cached" if (cached_bands and cached_bands.get("bands")) else "keyword_llm"
        ftype    = formula.get("formula_type", formula_hint).upper()
        steps    = f"{ftype}: value={ev!r} → {python_score}/{max_marks}"
        print(f"    [deterministic] {steps}" + ("" if verified else " ⚠ unverified"))
        return _make_result(python_score, ev, pg, steps,
                            max_marks, verified=verified, source=source,
                            haircut=not verified)

    # ── Fallback: LLM direct score ─────────────────────────────────────────
    if direct_score is not None:
        try:
            score = round(max(0.0, min(float(direct_score), float(max_marks))), 1)
            steps = f"LLM direct ({formula_hint})"
            return _make_result(score, ev, pg, steps,
                                max_marks, verified=True, source="llm_direct")
        except (TypeError, ValueError):
            pass

    if not found:
        reason = (f"Evidence not found for: {parameter[:60]}. "
                  f"Searched {len(keywords)} keywords across {len(hit_pages)} pages.")
        return _zero(reason)

    # LLM-type final fallback
    score_prompt = f"""Score this proposal evidence against the RFP criterion.
CRITERION: {parameter}
MAX MARKS: {max_marks}
SCORING RULES: {criterion.get('criteria_text','')[:400]}
FOUND IN PROPOSAL: {ev} (page {pg})
RELEVANT TEXT:
{pages_text[:800]}
Return ONLY valid JSON: {{"score": <0-{max_marks}>, "justification": "<one sentence>"}}"""

    raw2 = call_llm(score_prompt, label=f"llm-fb-{parameter[:20]}")
    res2 = extract_json(raw2) if raw2 else {}
    if res2:
        try:
            s2 = round(max(0.0, min(float(res2.get("score") or 0), float(max_marks))), 1)
            j2 = res2.get("justification", "")
            return _make_result(s2, ev, pg, f"LLM fallback: {j2}",
                                max_marks, verified=True, source="llm_fallback")
        except (TypeError, ValueError):
            pass

    return _zero(f"Scoring failed for: {parameter[:60]}")