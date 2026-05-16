"""
core/tq_proposal_scorer.py
===========================
Stage 3 — Score each criterion against the proposal.

Per criterion:
  1. Evidence retrieval  →  tq_evidence.get_proposal_pages()
  2. Fact extraction     →  tq_evidence.extract_fact() via Ollama
  3. Formula application →  tq_formula_engine.*
  4. Discrepancy check   →  year mismatch, wrong data, missing evidence
  5. Build result dict

For QUAL criteria: structured evidence check (no single-fact extraction).
For BINARY criteria: keyword presence check in proposal.
For LLM criteria:  bounded Ollama scoring call.
"""

from __future__ import annotations

import re
import requests
from pathlib import Path
from typing import Optional

from .tq_formula_engine import (
    apply_step, apply_band, apply_per_unit,
    apply_qual_structured, apply_binary, clamp,
)
from .tq_evidence import (
    extract_fact, extract_qual_evidence,
    get_proposal_pages, check_financial_years,
)

OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_MODEL    = "llama3.2"
SCORE_TIMEOUT   = 90

# Financial years that MOST Indian govt RFPs specify (last 3 ending current FY)
# The evaluator detects which years the proposal actually uses and compares.
COMMON_RFP_FY_PATTERNS = [
    ["2017-18", "2018-19", "2019-20"],
    ["2018-19", "2019-20", "2020-21"],
    ["2019-20", "2020-21", "2021-22"],
    ["2020-21", "2021-22", "2022-23"],
    ["2021-22", "2022-23", "2023-24"],
    ["2022-23", "2023-24", "2024-25"],
]


# ──────────────────────────────────────────────────────────────────────────────
# Detect required financial years from criteria_text
# ──────────────────────────────────────────────────────────────────────────────

def _detect_required_fy(criteria_text: str) -> list[str]:
    """
    Extract financial years mentioned explicitly in the RFP's criteria text.
    e.g. "(2017-18, 2018-19, 2019-20)" → ["2017-18", "2018-19", "2019-20"]
    """
    found = re.findall(r"\b(20\d{2}-\d{2,4})\b", criteria_text)
    # normalise "2019-2020" → "2019-20"
    normalised = []
    for fy in found:
        if len(fy) == 9 and "-" in fy:  # "2019-2020"
            normalised.append(fy[:5] + fy[7:])
        else:
            normalised.append(fy)
    return list(dict.fromkeys(normalised))  # deduplicate, preserve order


# ──────────────────────────────────────────────────────────────────────────────
# LLM scoring for complex criteria (methodology, approach quality)
# ──────────────────────────────────────────────────────────────────────────────

_LLM_SCORE_PROMPT = """\
Score a vendor proposal against one RFP evaluation criterion.

CRITERION NAME: {parameter}
MAX MARKS: {max_marks}
SCORING RULE: {rule}

EVIDENCE FOUND IN PROPOSAL:
Value: {value}
Evidence text: {evidence}

PROPOSAL EXCERPT:
{pages}

INSTRUCTIONS:
- Score 0 to {max_marks} based on quality, completeness, and specificity.
- If methodology/work plan is present and well-structured for this specific assignment: 60-80% of marks.
- If it is detailed, tailored to the RFP and demonstrates clear understanding: 80-100%.
- If it is generic, copied, or missing key components: 0-40%.
- Be strict — partial marks for partial evidence.

Return ONLY valid JSON:
{{"score": <0 to {max_marks}>, "justification": "one-sentence reason for score"}}"""


def _llm_score(
    parameter: str,
    criteria_text: str,
    max_marks: int,
    extracted: dict,
    proposal_path: str,
) -> tuple[float, str]:
    pages_text, _ = get_proposal_pages(
        proposal_path, parameter, criteria_text, "LLM", max_chars=2500
    )
    prompt = _LLM_SCORE_PROMPT.format(
        parameter=parameter,
        max_marks=max_marks,
        rule=criteria_text[:400],
        value=extracted.get("value") or "Not found",
        evidence=extracted.get("raw_evidence") or "No direct evidence",
        pages=pages_text[:1200],
    )
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096},
            },
            timeout=SCORE_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"] or ""
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        import json
        start = raw.find("{")
        if start >= 0:
            depth, in_str, esc = 0, False, False
            for i, ch in enumerate(raw[start:], start):
                if esc:            esc = False; continue
                if ch == "\\" and in_str: esc = True; continue
                if ch == '"':      in_str = not in_str; continue
                if in_str:         continue
                if ch == "{":      depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = re.sub(r",\s*([}\]])", r"\1", raw[start:i+1])
                        try:
                            d = json.loads(cand)
                            s = clamp(float(d.get("score") or 0), max_marks)
                            return s, d.get("justification", "")
                        except:
                            break
    except Exception as e:
        print(f"    [Scorer] LLM score error: {e}")

    # Regex fallback
    m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw if "raw" in dir() else "")
    if m:
        return clamp(float(m.group(1)), max_marks), "Score extracted by regex fallback"
    return 0.0, "LLM scoring failed — awarded 0"


# ──────────────────────────────────────────────────────────────────────────────
# Binary check — keyword presence in proposal
# ──────────────────────────────────────────────────────────────────────────────

def _binary_check(proposal_path: str, parameter: str, criteria_text: str) -> bool:
    """Check if the criterion is satisfied by keyword presence in proposal."""
    import fitz
    combined = (parameter + " " + criteria_text).lower()

    # Build target keywords from parameter
    targets: list[str] = []
    if "registered" in combined or "registration" in combined:
        targets += ["registered", "registration", "certificate of incorporation",
                    "cin", "llp", "company act", "societies act"]
    if "pan" in combined:
        targets.append("pan")
    if "gst" in combined:
        targets += ["gst", "gstin"]
    if "iso" in combined:
        targets += ["iso ", "iso-"]
    if "msme" in combined:
        targets += ["msme", "msmed", "udyam"]
    if "methodology" in combined or "approach" in combined:
        targets += ["methodology", "our approach", "proposed approach",
                    "technical approach", "work plan"]
    if not targets:
        return False

    doc = fitz.open(proposal_path)
    found = False
    for pno in range(min(len(doc), 200)):
        txt = doc[pno].get_text().lower()
        if any(t in txt for t in targets):
            found = True
            break
    doc.close()
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Result builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_result(
    score: float,
    max_marks: int,
    extracted: dict,
    steps: str,
    discrepancies: list[str],
    gaps: list[str],
) -> dict:
    score = clamp(score, max_marks)
    pct   = round((score / max_marks) * 100, 1) if max_marks else 0.0
    ev    = extracted.get("value") or "Not found"
    pg    = extracted.get("page")

    return {
        "score":           score,
        "score_percentage": pct,
        "extracted_value": ev,
        "source_page":     pg,
        "raw_evidence":    extracted.get("raw_evidence"),
        "pages_searched":  extracted.get("pages_searched", []),
        "scoring_steps":   steps,
        "justification":   (
            f"Score {score}/{max_marks} ({pct}%). Found: {ev}"
            + (f" (p.{pg})" if pg else "")
        ),
        "discrepancies":   discrepancies,
        "strengths":       [f"Found: {ev}" + (f" (p.{pg})" if pg else "")] if score > 0 else [],
        "gaps":            gaps,
        "evidence_found":  score > 0,
    }


def _zero(reason: str, pages: list[int] = None) -> dict:
    return {
        "score": 0, "score_percentage": 0.0,
        "extracted_value": None, "source_page": None,
        "raw_evidence": None, "pages_searched": pages or [],
        "scoring_steps": reason, "justification": reason,
        "discrepancies": [], "strengths": [], "gaps": [reason],
        "evidence_found": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main scorer
# ──────────────────────────────────────────────────────────────────────────────

def score_criterion(criterion: dict, proposal_path: str) -> dict:
    """
    Score one criterion against the proposal.

    criterion dict must have:
      parameter, max_marks, criteria_text, formula_hint

    Returns the standard result dict.
    """
    parameter     = criterion.get("parameter", "")
    max_marks     = int(criterion.get("max_marks") or 0)
    criteria_text = criterion.get("criteria_text", "")
    formula_hint  = criterion.get("formula_hint", "LLM")
    discrepancies: list[str] = []

    if max_marks == 0:
        return _zero("Zero-mark criterion")

    if not Path(proposal_path).exists():
        return _zero(f"Proposal file not found: {proposal_path}")

    print(f"    [Scorer] {parameter[:50]} | formula={formula_hint} | max={max_marks}")

    # ── QUAL: structured evidence check ──────────────────────────────────────
    if formula_hint == "QUAL":
        ev_dict = extract_qual_evidence(proposal_path, parameter, criteria_text)
        score, steps = apply_qual_structured(ev_dict, max_marks)
        gaps = []
        if score < max_marks * 0.8:
            missing = [k for k in ["named_experts", "education_stated",
                                    "experience_years_stated", "relevant_projects_listed",
                                    "cvs_attached"]
                       if not ev_dict.get(k)]
            gaps = [f"Missing: {', '.join(missing)}"] if missing else []
        return _build_result(
            score, max_marks,
            {"value": ev_dict.get("notes", "CV evidence check"),
             "page": None, "raw_evidence": None, "pages_searched": []},
            steps, discrepancies, gaps,
        )

    # ── BINARY: keyword presence check ───────────────────────────────────────
    if formula_hint == "BINARY":
        found = _binary_check(proposal_path, parameter, criteria_text)
        score, steps = apply_binary(found, max_marks, parameter)
        gaps = [f"'{parameter}' not evidenced in proposal"] if not found else []
        return _build_result(
            score, max_marks,
            {"value": "Yes" if found else "No", "page": None,
             "raw_evidence": None, "pages_searched": []},
            steps, discrepancies, gaps,
        )

    # ── Stage A: Extract the specific fact ───────────────────────────────────
    extracted = extract_fact(proposal_path, parameter, criteria_text, formula_hint)
    found     = extracted.get("found", False)
    ev        = extracted.get("value") or "Not found"

    print(f"    [Scorer] found={found} value={ev!r} page={extracted.get('page')}")

    # ── Financial year discrepancy check ─────────────────────────────────────
    required_fy = _detect_required_fy(criteria_text)
    used_fy     = extracted.get("financial_years_mentioned", [])
    if required_fy and used_fy:
        fy_check = check_financial_years(used_fy, required_fy)
        if fy_check.get("mismatch"):
            discrepancies.append(f"FINANCIAL YEAR MISMATCH: {fy_check['detail']}")

    # ── Stage B: Apply Python formula ────────────────────────────────────────
    if not found:
        if formula_hint == "LLM":
            # LLM scoring even if fact not explicitly found
            score, just = _llm_score(
                parameter, criteria_text, max_marks, extracted, proposal_path
            )
            return _build_result(
                score, max_marks, extracted, f"LLM: {just}",
                discrepancies,
                [f"Key fact not clearly stated: {parameter}"] if score < max_marks else [],
            )
        return _zero(
            f"Evidence not found in proposal for: {parameter}",
            extracted.get("pages_searched", []),
        )

    # STEP
    if formula_hint == "STEP":
        score, steps = apply_step(criteria_text, max_marks, ev)
        gaps = [] if score >= max_marks else [
            f"Turnover {ev} below maximum threshold for {max_marks} marks"
        ]
        return _build_result(score, max_marks, extracted, steps, discrepancies, gaps)

    # BAND
    if formula_hint == "BAND":
        score, steps = apply_band(criteria_text, max_marks, ev)
        gaps = [] if score >= max_marks else [
            f"Value {ev} below highest band threshold"
        ]
        return _build_result(score, max_marks, extracted, steps, discrepancies, gaps)

    # PER_UNIT
    if formula_hint == "PER_UNIT":
        score, steps = apply_per_unit(criteria_text, max_marks, ev)
        gaps = [] if score >= max_marks else [
            f"Additional qualifying projects needed for full {max_marks} marks"
        ]
        return _build_result(score, max_marks, extracted, steps, discrepancies, gaps)

    # LLM fallback
    score, just = _llm_score(
        parameter, criteria_text, max_marks, extracted, proposal_path
    )
    return _build_result(
        score, max_marks, extracted, f"LLM: {just}",
        discrepancies,
        [] if score >= max_marks else [f"Partial evidence for: {parameter}"],
    )
