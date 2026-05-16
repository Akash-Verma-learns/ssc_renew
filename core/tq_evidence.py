"""
core/tq_evidence.py
====================
Stage 2 — Retrieve the right evidence pages from the proposal for each criterion.

Knows WHAT to look for based on criterion type:
  - Turnover → CA certificate, audited balance sheet, financial summary pages
  - Projects → work orders, completion certificates, project description pages
  - Team/CV  → CV pages, team composition pages
  - Methodology → approach/methodology/work plan chapter

Also knows WHAT FACT to ask the LLM to find, and validates
financial years against RFP requirements.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_MODEL    = "llama3.2"
FACT_TIMEOUT    = 90     # per-criterion fact extraction timeout
MAX_PAGE_CHARS  = 3_000  # chars per page sent to LLM
MAX_PAGES_SENT  = 4      # max pages to send per criterion


# ──────────────────────────────────────────────────────────────────────────────
# Keyword sets per criterion type
# ──────────────────────────────────────────────────────────────────────────────

_KW: dict[str, list[str]] = {
    "turnover": [
        "turnover", "crore", "annual", "revenue", "balance sheet",
        "financial year", "average annual", "ca certificate",
        "chartered accountant", "audited", "advisory revenue",
        "advisory services", "gross receipts",
    ],
    "projects": [
        "project", "assignment", "work order", "completion certificate",
        "experience certificate", "pmc", "pmu", "urban", "amrut",
        "smart city", "client billing", "contract value", "agreement",
        "letter of award", "mandate",
    ],
    "pma_pmu": [
        "pma", "pmu", "program management", "project management unit",
        "programme management", "scheme implementation",
        "central government", "state government", "multilateral",
        "ministry", "department", "mofpi", "midh", "nhb", "sfac",
    ],
    "professionals": [
        "professional", "manpower", "employees", "supply", "deployed",
        "staffing", "single order", "headcount", "consultants",
        "advisory staff", "number of", "advisory professionals",
    ],
    "qualification": [
        "cv", "curriculum vitae", "qualification", "years of experience",
        "team leader", "expert", "education", "degree", "relevant experience",
        "proposed team", "resume", "post graduate", "mba", "b.tech",
    ],
    "methodology": [
        "methodology", "approach", "work plan", "implementation",
        "technical approach", "strategy", "plan of action",
        "proposed approach", "our methodology",
    ],
    "registration": [
        "registered", "registration", "certificate of incorporation",
        "pan", "gst", "cin", "llp", "company", "society", "trust",
        "msme", "startup india", "iso",
    ],
    "generic": [
        "experience", "project", "relevant", "work", "assignment",
        "client", "contract",
    ],
}

# Map formula_hint → keyword set name
_FORMULA_TO_KW = {
    "STEP":     "turnover",
    "BAND":     "professionals",
    "PER_UNIT": "projects",
    "QUAL":     "qualification",
    "BINARY":   "registration",
    "LLM":      "methodology",
}


# ──────────────────────────────────────────────────────────────────────────────
# What fact to extract per formula type
# ──────────────────────────────────────────────────────────────────────────────

def _what_to_find(parameter: str, criteria_text: str, formula_hint: str) -> str:
    combined = (parameter + " " + criteria_text).lower()

    if formula_hint in ("STEP", "BAND") and any(
        w in combined for w in ["turnover", "crore", "revenue", "annual"]
    ):
        return (
            "The bidder's AVERAGE ANNUAL TURNOVER in INR Crore over the "
            "last 3 financial years, as stated in their CA certificate or "
            "audited balance sheet. Give a single number in Crore (e.g. '180 Cr'). "
            "Also note which financial years are mentioned."
        )
    if formula_hint == "BAND" and any(
        w in combined for w in ["professional", "manpower", "employee", "supply"]
    ):
        return (
            "The MAXIMUM number of professionals / employees supplied / deployed "
            "in a SINGLE work order for advisory or consulting services. "
            "Give a single integer count (e.g. '12 professionals')."
        )
    if formula_hint in ("PER_UNIT", "LLM") and any(
        w in combined for w in ["project", "assignment", "pmc", "pmu", "experience", "work order"]
    ):
        return (
            "The NUMBER of qualifying consulting / PMC / PMU / advisory projects "
            "cited in the proposal. Each project must have a named client, "
            "contract value, and duration. Give a single integer count "
            "(e.g. '5 projects')."
        )
    if formula_hint == "QUAL":
        return (
            "For each proposed team member: their name, role/position, "
            "highest educational qualification (degree + institution), "
            "and total years of relevant experience. List ALL members."
        )
    if formula_hint == "BINARY":
        return (
            "Whether the proposal / bidder meets the specific requirement: "
            f"'{parameter}'. Answer YES or NO with the evidence found."
        )
    if formula_hint == "LLM":
        return (
            f"Any relevant information about: {parameter}. "
            "Describe what the proposal says about this criterion."
        )
    return f"The specific information required by the criterion: {parameter}"


# ──────────────────────────────────────────────────────────────────────────────
# Page scoring (keyword hit count, no embeddings)
# ──────────────────────────────────────────────────────────────────────────────

def _pick_kw_set(parameter: str, criteria_text: str, formula_hint: str) -> list[str]:
    """Return the best keyword set for this criterion."""
    combined = (parameter + " " + criteria_text).lower()

    # PMA/PMU experience — special case
    if any(w in combined for w in ["pma", "pmu", "program management unit",
                                    "project management unit"]):
        return _KW["pma_pmu"] + _KW["projects"]

    # Turnover
    if any(w in combined for w in ["turnover", "revenue", "crore"]):
        return _KW["turnover"]

    # Manpower/professionals
    if any(w in combined for w in ["professional", "manpower", "employee"]):
        return _KW["professionals"]

    # Projects / assignments
    if any(w in combined for w in ["project", "assignment", "work order", "pmc", "pmu"]):
        return _KW["projects"]

    # Qualifications
    if any(w in combined for w in ["qualification", "competence", "cv", "curriculum"]):
        return _KW["qualification"]

    # Methodology
    if any(w in combined for w in ["methodology", "approach", "work plan"]):
        return _KW["methodology"]

    # Registration / certification
    if any(w in combined for w in ["registered", "certified", "accredited"]):
        return _KW["registration"]

    # Formula hint fallback
    kw_name = _FORMULA_TO_KW.get(formula_hint, "generic")
    return _KW.get(kw_name, _KW["generic"])


def get_proposal_pages(
    proposal_path: str,
    parameter: str,
    criteria_text: str,
    formula_hint: str,
    max_chars: int = MAX_PAGE_CHARS * MAX_PAGES_SENT,
    max_pages: int = MAX_PAGES_SENT,
) -> tuple[str, list[int]]:
    """
    Open the proposal PDF directly, score every page by keyword hits,
    return (concatenated_text, page_numbers_used).
    """
    kws = _pick_kw_set(parameter, criteria_text, formula_hint)

    doc = fitz.open(proposal_path)
    scored: list[tuple[int, int, str]] = []   # (hits, page_no_1based, text)

    for pno in range(len(doc)):
        txt = doc[pno].get_text()
        low = txt.lower()
        hits = sum(1 for kw in kws if kw in low)
        if hits > 0:
            scored.append((hits, pno + 1, txt.strip()))

    scored.sort(reverse=True)
    doc.close()

    if not scored:
        # Fallback: first 8 pages (cover letter / executive summary)
        doc2 = fitz.open(proposal_path)
        parts = []
        for i in range(min(8, len(doc2))):
            t = doc2[i].get_text().strip()
            if t:
                parts.append(f"[Page {i+1}]\n{t[:MAX_PAGE_CHARS]}")
        doc2.close()
        text = "\n\n---\n\n".join(parts)[:max_chars]
        page_nums = list(range(1, min(9, len(parts) + 1)))
        return text, page_nums

    parts, total, page_nums = [], 0, []
    for _, pno, txt in scored[:max_pages]:
        block = f"[Page {pno}]\n{txt[:MAX_PAGE_CHARS]}"
        if total + len(block) > max_chars:
            block = block[:max_chars - total]
        parts.append(block)
        page_nums.append(pno)
        total += len(block)
        if total >= max_chars:
            break

    return "\n\n---\n\n".join(parts), page_nums


# ──────────────────────────────────────────────────────────────────────────────
# Ollama fact extraction
# ──────────────────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Read the proposal pages below and find ONE specific piece of information.

FIND: {what}

PROPOSAL PAGES:
{pages}

Return ONLY valid JSON — nothing else:
{{
  "found": true or false,
  "value": "<exact value — e.g. '180 Cr' or '8 professionals' or '5 projects'>",
  "financial_years_mentioned": ["2017-18", "2018-19", "2019-20"],
  "page": <integer page number or null>,
  "raw_evidence": "<verbatim sentence or phrase from proposal>"
}}

If the information is not found: {{"found": false, "value": null, "financial_years_mentioned": [], "page": null, "raw_evidence": null}}"""


def _ollama_fact(prompt: str) -> Optional[dict]:
    import requests, json
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096},
            },
            timeout=FACT_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"] or ""
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        start = raw.find("{")
        if start < 0:
            return None
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
                    try: return json.loads(cand)
                    except: return None
        return None
    except Exception as e:
        print(f"[Evidence] Ollama error: {e}")
        return None


def extract_fact(
    proposal_path: str,
    parameter: str,
    criteria_text: str,
    formula_hint: str,
) -> dict:
    """
    Returns:
    {
        "found": bool,
        "value": str | None,
        "financial_years_mentioned": list[str],
        "page": int | None,
        "raw_evidence": str | None,
        "pages_searched": list[int],
    }
    """
    pages_text, page_nums = get_proposal_pages(
        proposal_path, parameter, criteria_text, formula_hint
    )

    if not pages_text.strip():
        return _not_found(page_nums)

    what = _what_to_find(parameter, criteria_text, formula_hint)
    prompt = _EXTRACT_PROMPT.format(what=what, pages=pages_text)
    print(f"    [Evidence] fact-extract: {len(prompt)} chars, pages={page_nums[:6]}")

    result = _ollama_fact(prompt)
    if not result:
        return _not_found(page_nums)

    return {
        "found":                    bool(result.get("found")),
        "value":                    str(result["value"]) if result.get("value") else None,
        "financial_years_mentioned": result.get("financial_years_mentioned") or [],
        "page":                     result.get("page"),
        "raw_evidence":             result.get("raw_evidence"),
        "pages_searched":           page_nums,
    }


def _not_found(pages: list[int]) -> dict:
    return {
        "found": False, "value": None,
        "financial_years_mentioned": [], "page": None,
        "raw_evidence": None, "pages_searched": pages,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Financial year validation
# ──────────────────────────────────────────────────────────────────────────────

def check_financial_years(
    years_mentioned: list[str],
    years_required: list[str],
) -> dict:
    """
    Compare years used in proposal vs years required by RFP.
    Returns a discrepancy dict.
    """
    if not years_required or not years_mentioned:
        return {"mismatch": False, "detail": ""}

    req_set  = set(years_required)
    used_set = set(years_mentioned)
    missing  = sorted(req_set - used_set)
    extra    = sorted(used_set - req_set)

    if missing or extra:
        return {
            "mismatch": True,
            "required":  sorted(years_required),
            "used":      sorted(years_mentioned),
            "missing":   missing,
            "extra":     extra,
            "detail": (
                f"RFP requires years {sorted(years_required)}, "
                f"proposal uses {sorted(years_mentioned)}. "
                f"Missing: {missing}. Extra: {extra}."
            ),
        }
    return {"mismatch": False, "detail": "Years match RFP requirement."}


# ──────────────────────────────────────────────────────────────────────────────
# Qualification evidence check (for QUAL formula)
# ──────────────────────────────────────────────────────────────────────────────

_QUAL_PROMPT = """\
Read the proposal pages below. For each question, answer YES or NO.

PROPOSAL PAGES:
{pages}

Questions:
1. Are specific team members / experts NAMED (not just role titles)?
2. Is the EDUCATIONAL QUALIFICATION stated for each expert (degree + field)?
3. Are YEARS OF RELEVANT EXPERIENCE stated for each expert?
4. Are RELEVANT PROJECTS / ASSIGNMENTS listed for the proposed experts?
5. Are CVs or detailed professional profiles ATTACHED or described?

Return ONLY valid JSON:
{{
  "named_experts":           true or false,
  "education_stated":        true or false,
  "experience_years_stated": true or false,
  "relevant_projects_listed": true or false,
  "cvs_attached":            true or false,
  "notes": "one sentence summary of findings"
}}"""


def extract_qual_evidence(
    proposal_path: str,
    parameter: str,
    criteria_text: str,
) -> dict:
    """
    Returns evidence dict for QUAL formula:
    {named_experts, education_stated, experience_years_stated,
     relevant_projects_listed, cvs_attached, notes}
    """
    import requests, json as _json

    pages_text, _ = get_proposal_pages(
        proposal_path, parameter, criteria_text, "QUAL", max_chars=4000
    )
    if not pages_text.strip():
        return _qual_empty()

    prompt = _QUAL_PROMPT.format(pages=pages_text[:3500])
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096},
            },
            timeout=FACT_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"] or ""
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        start = raw.find("{")
        if start < 0:
            return _qual_empty()
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
                        return _json.loads(cand)
                    except:
                        return _qual_empty()
        return _qual_empty()
    except Exception as e:
        print(f"[Evidence] QUAL extract error: {e}")
        return _qual_empty()


def _qual_empty() -> dict:
    return {
        "named_experts": False, "education_stated": False,
        "experience_years_stated": False, "relevant_projects_listed": False,
        "cvs_attached": False, "notes": "Could not extract qualification evidence.",
    }
