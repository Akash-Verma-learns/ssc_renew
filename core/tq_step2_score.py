"""
tq_step2_score.py  —  Python-formula scoring with tiny LLM fact extraction
===========================================================================

Design principles
-----------------
1. LLM is used ONLY to extract ONE specific fact per criterion.
   The prompt is always < 2 000 chars.  Timeout = 60 s.
   No arithmetic, no judgment — just "find this number in the text".

2. ALL scoring arithmetic is done in Python.
   No LLM rounding errors, no band misreads, no percentage confusion.

3. Proposal pages are ranked by keyword hit count (PyMuPDF direct, not ChromaDB).
   Top 3 pages, max 2 000 chars, sent to the LLM.

4. Formula types (auto-detected from criteria_text):
     STEP       Turnover: base + increments  (e.g. 100Cr=5, +0.5 per 10Cr)
     BAND       Professionals: ordered threshold bands (6=10, 7-12=15, >12=20)
     PER_UNIT   Projects: N marks per project, capped at max  (5 per proj, max 20)
     QUAL       Qualifications: structured yes/no evidence check, small LLM
     LLM        Methodology and all others: bounded LLM call (< 1 500 chars)

5. Timeout guard: every Ollama call is wrapped; returns score=0 on timeout
   rather than hanging the whole pipeline.
"""

from __future__ import annotations

import json
import re
import requests
from pathlib import Path
from typing import Optional, Callable

try:
    import fitz
except ImportError:
    raise ImportError("pip install pymupdf")

OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"

# Timeout per Ollama call — small prompts should finish in < 30 s on CPU
OLLAMA_TIMEOUT_SHORT = 60    # fact extraction (< 2 000 char prompt)
OLLAMA_TIMEOUT_SCORE = 90    # scoring calls (< 1 500 char prompt)

TQ_UPLOAD_DIR = Path("./tq_uploads")


# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ollama(prompt: str, timeout: int = OLLAMA_TIMEOUT_SHORT) -> str:
    """Call Ollama with a hard timeout.  Returns "" on any failure."""
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.0, "num_ctx": 4096},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"] or ""
    except requests.exceptions.Timeout:
        print(f"[Step2] Ollama timeout after {timeout}s (prompt={len(prompt)} chars)")
        return ""
    except Exception as e:
        print(f"[Step2] Ollama error: {e}")
        return ""


def _parse_json(text: str) -> Optional[dict]:
    """Extract first complete JSON object from LLM output."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    start = text.find("{")
    if start < 0:
        return None

    depth = in_str = esc = False
    depth_n = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if esc:            esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"':      in_str = not in_str; continue
        if in_str:         continue
        if ch == "{":      depth_n += 1
        elif ch == "}":
            depth_n -= 1
            if depth_n == 0:
                end = i + 1
                break

    candidate = text[start:end] if end > start else text[start:]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _regex_score(text: str, max_marks: int) -> Optional[float]:
    m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
    if not m:
        m = re.search(r'\bscore\b.*?(\d+(?:\.\d+)?)', text, re.I)
    if m:
        v = float(m.group(1))
        return round(v, 1) if 0 <= v <= max_marks else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Proposal page ranking (keyword hit count)
# ─────────────────────────────────────────────────────────────────────────────

_KW_SETS = {
    "turnover":     ["turnover", "crore", "annual", "revenue", "balance sheet",
                     "financial year", "average annual", "gross turnover"],
    "professionals":["professional", "manpower", "order", "employees", "advisory",
                     "consulting", "supply", "deployed", "staffing", "single order"],
    "projects":     ["project", "assignment", "pmc", "pmu", "urban", "amrut",
                     "smart city", "completion certificate", "ulb", "billing",
                     "work order", "experience certificate"],
    "qualification":["cv", "curriculum vitae", "qualification", "years of experience",
                     "team leader", "expert", "education", "degree", "b.tech",
                     "relevant experience", "proposed team", "resume"],
    "methodology":  ["methodology", "approach", "work plan", "implementation",
                     "technical approach", "pmu assignment", "strategy", "plan"],
    "generic":      ["experience", "project", "relevant", "work", "assignment"],
}


def _pick_kw_set(parameter: str, criteria_text: str) -> list[str]:
    combined = (parameter + " " + criteria_text).lower()
    if any(w in combined for w in ["turnover", "crore", "annual"]):
        return _KW_SETS["turnover"]
    if any(w in combined for w in ["professional", "manpower", "employees in supply",
                                    "number of employees"]):
        return _KW_SETS["professionals"]
    if any(w in combined for w in ["consulting", "pmc", "pmu", "urban",
                                    "amrut", "billing", "assignment"]):
        return _KW_SETS["projects"]
    if any(w in combined for w in ["qualification", "competence", "cv",
                                    "curriculum", "staff", "proposed team"]):
        return _KW_SETS["qualification"]
    if any(w in combined for w in ["methodology", "approach", "work plan"]):
        return _KW_SETS["methodology"]
    return _KW_SETS["generic"]


def _get_proposal_pages(
    proposal_path: str,
    parameter: str,
    criteria_text: str,
    max_chars: int = 2_000,
    max_pages: int = 3,
) -> str:
    """
    Open the proposal PDF directly with PyMuPDF.
    Score every page by keyword hits; return top pages as text.
    """
    kws    = _pick_kw_set(parameter, criteria_text)
    doc    = fitz.open(proposal_path)
    scored = []
    for pno in range(len(doc)):
        txt  = doc[pno].get_text()
        low  = txt.lower()
        hits = sum(1 for kw in kws if kw in low)
        if hits > 0:
            scored.append((hits, pno + 1, txt.strip()))

    scored.sort(reverse=True)
    doc.close()

    if not scored:
        # Fallback: first 8 pages
        doc2   = fitz.open(proposal_path)
        parts  = [f"[Page {i+1}]\n{doc2[i].get_text().strip()[:300]}"
                  for i in range(min(8, len(doc2)))]
        doc2.close()
        return "\n\n".join(parts)[:max_chars]

    parts, total = [], 0
    for _, pno, txt in scored[:max_pages]:
        block = f"[Page {pno}]\n{txt}"
        if total + len(block) > max_chars:
            block = block[:max_chars - total]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Fact extraction (Stage A) — TINY prompt, one specific question
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Read the proposal pages below and find ONE specific piece of information.

FIND: {what}

PROPOSAL PAGES:
{pages}

Return ONLY valid JSON — nothing else:
{{"found": true or false, "value": "<exact value, e.g. 180 Cr or 8 professionals or 3 projects>", "page": <integer page number or null>}}

If not found: {{"found": false, "value": null, "page": null}}"""


def _what_to_find(parameter: str, criteria_text: str) -> str:
    combined = (parameter + " " + criteria_text).lower()
    if "turnover" in combined or ("crore" in combined and "turnover" in combined):
        return ("the bidder's average annual turnover in Indian Rupees Crore (INR Cr) "
                "over the last 3 financial years — give a single number in Crore")
    if any(w in combined for w in ["professional", "manpower", "employees in supply",
                                    "number of employees"]):
        return ("the maximum number of professionals / employees supplied / deployed "
                "in a SINGLE work order for advisory or consulting services — "
                "give a single integer count")
    if any(w in combined for w in ["pmc", "pmu", "urban", "amrut", "billing", "assignment"]):
        return ("the number of eligible consulting / advisory / PMC / PMU projects "
                "where the client billing is at least Rs 0.4 Cr per assignment — "
                "give a single integer project count")
    if any(w in combined for w in ["qualification", "competence", "cv", "curriculum"]):
        return ("for each proposed team member or expert: their name, role, "
                "highest educational qualification, and years of relevant experience")
    if any(w in combined for w in ["methodology", "approach", "work plan"]):
        return ("whether the proposal contains a technical methodology or work plan "
                "for the assignment — answer YES with page reference, or NO")
    return f"relevant information for the criterion: {parameter}"


def _extract_fact(proposal_path: str, parameter: str, criteria_text: str) -> dict:
    """Returns {found: bool, value: str|None, page: int|None}"""
    pages  = _get_proposal_pages(proposal_path, parameter, criteria_text)
    what   = _what_to_find(parameter, criteria_text)
    prompt = _EXTRACT_PROMPT.format(what=what, pages=pages)

    print(f"    [A] Fact-extract prompt: {len(prompt)} chars")
    raw    = _ollama(prompt, timeout=OLLAMA_TIMEOUT_SHORT)
    result = _parse_json(raw) if raw else None
    if result:
        return {
            "found": bool(result.get("found")),
            "value": str(result["value"]) if result.get("value") else None,
            "page":  result.get("page"),
        }
    return {"found": False, "value": None, "page": None}


# ─────────────────────────────────────────────────────────────────────────────
# Formula detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_formula_type(parameter: str, criteria_text: str) -> str:
    ct = criteria_text.lower()
    p  = parameter.lower()

    # Step scoring: base value + increments (turnover)
    if re.search(r'every\s+additional\s+\d+\s*cr[ore]*', ct, re.I):
        return "STEP"

    # Band scoring: multiple explicit threshold bands for professionals
    if len(re.findall(r'\d+\s*professional', ct, re.I)) >= 2:
        return "BAND"

    # Per-unit with cap: N marks per project/assignment, maximum M marks
    if re.search(r'\d+\s*marks?\s+for\s+(?:\d+|each|01|per)\s+project', ct, re.I):
        return "PER_UNIT"

    # Qualifications: percentage-weighted multi-component
    if re.search(r'(\d+%.*?(education|experience|project|qualification))', ct, re.I):
        return "QUAL"
    if "qualification" in p and "competence" in p:
        return "QUAL"

    return "LLM"


# ─────────────────────────────────────────────────────────────────────────────
# Python formula implementations
# ─────────────────────────────────────────────────────────────────────────────

def _apply_step_formula(criteria_text: str, max_marks: int, value_str: str) -> Optional[float]:
    """
    Turnover step scoring:
      "Turnover 100 Cr = 5 marks. For every additional 10 Cr = 0.5 marks"
    → score = base_marks + floor((turnover - base) / step) × step_marks
    → capped at max_marks
    """
    ct = criteria_text

    # Parse base: "100 Cr = 5 marks" or "100 Cr.=5 marks" or "Turnover -100 Cr. =5 marks"
    base_m = re.search(
        r'(\d+(?:\.\d+)?)\s*cr[ores]*\s*[.=:\-–\s]+\s*(\d+(?:\.\d+)?)\s*marks?',
        ct, re.IGNORECASE,
    )
    # Parse step: "every additional 10 Cr = 0.5 marks" or "+0.5 per 10 Cr"
    step_m = re.search(
        r'(?:every|each|per)\s+additional\s+(\d+(?:\.\d+)?)\s*cr[ores]*'
        r'[\s\W]*(\d+(?:\.\d+)?)\s*marks?',
        ct, re.IGNORECASE,
    )
    if not (base_m and step_m):
        return None

    try:
        base_threshold = float(base_m.group(1))
        base_score     = float(base_m.group(2))
        step_size      = float(step_m.group(1))
        step_score     = float(step_m.group(2))

        # Extract numeric value from the found string (e.g. "180 Cr" → 180)
        nums = re.findall(r'[\d,]+(?:\.\d+)?', value_str.replace(",", ""))
        if not nums:
            return None
        turnover = float(nums[0])

        if turnover < base_threshold:
            return 0.0

        extra_steps = int((turnover - base_threshold) / step_size)
        score = base_score + extra_steps * step_score
        return round(min(score, max_marks), 1)

    except (ValueError, ZeroDivisionError):
        return None


def _apply_band_formula(criteria_text: str, max_marks: int, value_str: str) -> Optional[float]:
    """
    Band scoring for professionals:
      "06 professionals : 10 marks"
      "more than 06 and up to 12 professionals : 15 marks"
      "more than 12 professionals : 20 marks"
    """
    ct = criteria_text.lower()

    # Build a list of (upper_bound, marks) tuples
    # Simple pattern: "N professionals : M marks"
    simple_bands = re.findall(
        r'(?:of\s+)?(\d+)\s+professionals?\s*[:\-–]\s*(\d+)\s*marks?',
        ct, re.IGNORECASE,
    )
    # Range pattern: "more than N1[- and up to N2] professionals : M marks"
    range_bands = re.findall(
        r'more\s+than\s+(\d+)(?:[\s\-–]*and[\s\-–]*up[\s\-–]*to\s+(\d+))?\s+'
        r'professionals?\s*[:\-–]\s*(\d+)\s*marks?',
        ct, re.IGNORECASE,
    )

    bands = []
    for count_str, marks_str in simple_bands:
        bands.append((int(count_str), int(marks_str)))
    for lo_str, hi_str, marks_str in range_bands:
        upper = int(hi_str) if hi_str else 9999
        bands.append((upper, int(marks_str)))

    if not bands:
        return None

    bands.sort(key=lambda b: b[0])

    try:
        nums = re.findall(r'\d+', value_str)
        if not nums:
            return None
        count = int(nums[0])

        score = 0.0
        for upper, marks in bands:
            if count <= upper:
                score = float(marks)
                break
        else:
            # count > all defined thresholds → take the max band score
            score = float(bands[-1][1])

        return round(min(score, max_marks), 1)

    except (ValueError, IndexError):
        return None


def _apply_per_unit_formula(criteria_text: str, max_marks: int, value_str: str) -> Optional[float]:
    """
    Per-unit scoring:
      "5 marks for 01 project with maximum of 20 marks"
    → score = min(count × rate, max_marks)
    """
    ct = criteria_text.lower()

    rate_m = re.search(
        r'(\d+(?:\.\d+)?)\s*marks?\s+for\s+(?:\d+|each|01|per)\s+(?:project|assignment)',
        ct, re.IGNORECASE,
    )
    if not rate_m:
        return None

    try:
        rate = float(rate_m.group(1))
        nums = re.findall(r'\d+', value_str)
        if not nums:
            return None
        count = int(nums[0])
        return round(min(count * rate, max_marks), 1)
    except (ValueError, ZeroDivisionError):
        return None


def _apply_qual_formula(
    proposal_path: str,
    parameter: str,
    criteria_text: str,
    max_marks: int,
) -> tuple[float, str]:
    """
    Qualifications / competence scoring.
    Uses a structured binary evidence check — very small LLM call.
    """
    pages = _get_proposal_pages(proposal_path, parameter, criteria_text)

    prompt = f"""Check if the proposal includes the following. Answer YES or NO for each.

1. Named team members or proposed experts?
2. Educational qualifications stated (degree, diploma, certification)?
3. Years of relevant experience stated for each expert?
4. Relevant projects or assignments listed for the proposed team?
5. CVs or detailed profiles attached?

PROPOSAL (read carefully):
{pages[:1500]}

Return ONLY valid JSON:
{{"q1": true/false, "q2": true/false, "q3": true/false, "q4": true/false, "q5": true/false, "note": "one short sentence"}}"""

    print(f"    [Qual] Prompt: {len(prompt)} chars")
    raw    = _ollama(prompt, timeout=OLLAMA_TIMEOUT_SCORE)
    result = _parse_json(raw) if raw else {}

    if not result:
        return 0.0, "Qualification check failed (LLM timeout or parse error)"

    # Weighted scoring: evidence drives marks
    weights = {"q1": 0.10, "q2": 0.20, "q3": 0.20, "q4": 0.30, "q5": 0.20}
    total_w = sum(w for k, w in weights.items() if result.get(k, False))
    score   = round(total_w * max_marks, 1)
    note    = result.get("note", "")
    return score, note


def _apply_llm_formula(
    proposal_path: str,
    parameter: str,
    criteria_text: str,
    max_marks: int,
    extracted: dict,
) -> tuple[float, str]:
    """
    LLM scoring for complex criteria (methodology, etc.)
    Prompt is strictly bounded to < 1 500 chars.
    """
    pages = _get_proposal_pages(proposal_path, parameter, criteria_text, max_chars=1_200)
    rule  = criteria_text[:300]
    value = extracted.get("value") or "Not explicitly found in proposal"
    pg    = extracted.get("page") or "unknown"

    prompt = f"""Score a vendor proposal against one RFP criterion.

CRITERION: {parameter}
MAX MARKS: {max_marks}
SCORING RULE: {rule}

WHAT WAS FOUND: {value} (page {pg})

PROPOSAL EXCERPT:
{pages[:800]}

Instructions:
- Award marks based on the quality and relevance of the evidence.
- If methodology/work plan is present and well-structured, award 60-80% of marks.
- If it is detailed and specifically tailored to the RFP, award 80-100%.
- If absent or generic, award 0-30%.
- Score must be 0 to {max_marks}.

Return ONLY valid JSON:
{{"score": <0 to {max_marks}>, "justification": "one sentence"}}"""

    print(f"    [LLM] Prompt: {len(prompt)} chars")
    raw    = _ollama(prompt, timeout=OLLAMA_TIMEOUT_SCORE)
    result = _parse_json(raw) if raw else {}

    if not result:
        fallback = _regex_score(raw or "", max_marks)
        return (fallback or 0.0), "LLM parse failed"

    try:
        score = round(max(0.0, min(float(result.get("score") or 0), float(max_marks))), 1)
        return score, result.get("justification", "")
    except (TypeError, ValueError):
        return 0.0, "Score conversion failed"


# ─────────────────────────────────────────────────────────────────────────────
# Score one criterion
# ─────────────────────────────────────────────────────────────────────────────

def score_criterion(criterion: dict, proposal_path: str) -> dict:
    """
    Two stages:
      A. LLM extracts the specific fact (tiny prompt, 60s timeout).
      B. Python formula computes the score from that fact.
         Falls back to bounded LLM scoring for complex criteria.
    """
    max_marks     = int(criterion.get("max_marks") or 0)
    parameter     = criterion.get("parameter", "")
    criteria_text = criterion.get("criteria_text", "")

    if max_marks == 0:
        return _zero("Zero-mark criterion")

    if not Path(proposal_path).exists():
        return _zero(f"Proposal file not found: {proposal_path}")

    formula_type = _detect_formula_type(parameter, criteria_text)
    print(f"    [formula] type={formula_type}")

    # ── Qualifications: structured evidence check, no single-fact extraction ──
    if formula_type == "QUAL":
        score, note = _apply_qual_formula(proposal_path, parameter, criteria_text, max_marks)
        return {
            "score":           score,
            "extracted_value": note or "Qualifications evidence check",
            "source_page":     None,
            "scoring_steps":   f"Structured evidence check → {score}/{max_marks}",
            "justification":   note or f"Score {score}/{max_marks} based on CV evidence",
            "strengths":       [note] if note and score > 0 else [],
            "gaps":            [] if score >= max_marks * 0.8 else ["Full marks require detailed CVs for all roles"],
            "evidence_found":  score > 0,
        }

    # ── Stage A: Extract the specific fact ───────────────────────────────────
    extracted = _extract_fact(proposal_path, parameter, criteria_text)
    ev        = extracted.get("value") or "Not found"
    pg        = extracted.get("page")
    found     = extracted.get("found", False)
    print(f"    [A] found={found}  value={ev!r}  page={pg}")

    if not found:
        # LLM fallback even if fact not found — might still award partial marks
        if formula_type == "LLM":
            score, just = _apply_llm_formula(
                proposal_path, parameter, criteria_text, max_marks, extracted
            )
            return _result(score, ev, pg, f"LLM scoring: {just}", max_marks)
        return _zero(f"Key fact not found: {_what_to_find(parameter, criteria_text)[:80]}")

    # ── Stage B: Apply Python formula ────────────────────────────────────────
    python_score: Optional[float] = None

    if formula_type == "STEP":
        python_score = _apply_step_formula(criteria_text, max_marks, ev)
        steps = f"STEP formula: value={ev} → {python_score}/{max_marks}"

    elif formula_type == "BAND":
        python_score = _apply_band_formula(criteria_text, max_marks, ev)
        steps = f"BAND formula: value={ev} → {python_score}/{max_marks}"

    elif formula_type == "PER_UNIT":
        python_score = _apply_per_unit_formula(criteria_text, max_marks, ev)
        steps = f"PER_UNIT formula: value={ev} → {python_score}/{max_marks}"

    else:
        # LLM scoring (methodology, etc.)
        score, just = _apply_llm_formula(
            proposal_path, parameter, criteria_text, max_marks, extracted
        )
        return _result(score, ev, pg, f"LLM: {just}", max_marks)

    if python_score is not None:
        print(f"    [B-Python] {steps}")
        return _result(python_score, ev, pg, steps, max_marks)

    # Python parse failed for a formula we thought we could handle → LLM fallback
    print(f"    [B-LLM fallback] Python formula parse failed for {formula_type}")
    score, just = _apply_llm_formula(
        proposal_path, parameter, criteria_text, max_marks, extracted
    )
    return _result(score, ev, pg, f"LLM fallback (formula parse failed): {just}", max_marks)


def _result(score: float, ev: str, pg: Optional[int], steps: str, max_marks: int) -> dict:
    score = round(max(0.0, min(score, float(max_marks))), 1)
    return {
        "score":           score,
        "extracted_value": ev,
        "source_page":     pg,
        "scoring_steps":   steps,
        "justification":   f"Score {score}/{max_marks}. Found: {ev}" + (f" (p.{pg})" if pg else ""),
        "strengths":       [f"Found: {ev}" + (f" (p.{pg})" if pg else "")] if score > 0 else [],
        "gaps":            [] if score >= max_marks else [f"Full marks require additional evidence"],
        "evidence_found":  score > 0,
    }


def _zero(reason: str) -> dict:
    return {
        "score": 0, "extracted_value": None, "source_page": None,
        "scoring_steps": reason, "justification": reason,
        "strengths": [], "gaps": [reason], "evidence_found": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Score all criteria
# ─────────────────────────────────────────────────────────────────────────────

def score_all_criteria(
    criteria:          list,
    proposal_path:     str,
    progress_callback: Optional[Callable] = None,
) -> list:
    n      = len(criteria)
    scores = []

    for i, criterion in enumerate(criteria):
        pct  = 28 + int((i / max(n, 1)) * 65)
        step = f"Scoring: {criterion.get('parameter', '')[:55]}"
        if progress_callback:
            try: progress_callback(step, pct)
            except Exception: pass
        print(f"[Step2] {pct:3d}% -- {step}")

        try:
            result = score_criterion(criterion, proposal_path)
        except Exception as e:
            print(f"[Step2] Exception: {e}")
            result = _zero(str(e))

        s  = result.get("score", 0)
        pg = f"(p.{result['source_page']})" if result.get("source_page") else ""
        print(f"  [{i+1}/{n}] {criterion.get('parameter','')[:55]:55s} "
              f"-> {s}/{criterion.get('max_marks',0)} {pg}")

        scores.append({
            "item_code":                       criterion.get("item_code", str(i + 1)),
            "parameter":                       criterion.get("parameter", ""),
            "max_marks":                       criterion.get("max_marks", 0),
            "criteria_text":                   criterion.get("criteria_text", ""),
            "is_sub_item":                     False,
            "parent_parameter":                "",
            "evaluation_layer":                "document",
            "requires_live_assessment":        False,
            "requires_comparative_evaluation": False,
            **result,
        })

    return scores