"""
core/tq_extractor_patch.py — Repair-Pass Hallucination Fix
============================================================

BUG: When criteria sum to less than doc_total, the repair pass asks the LLM
to "find the missing marks."  The LLM invents a plausible-sounding criterion
("Additional Experience: 30 marks") rather than recognising that the gap is
intentional (e.g. the grand total includes live marks that were already
subtracted, OR the RFP genuinely has fewer marks than expected).

EXAMPLE (from the DDU-GKY Pune RFP):
  grand_total = 130
  live_marks  = 30  (Technical Presentation)
  doc_total   = 100  ← correct
  criteria sum = 70  ← RFP only has 5 criteria summing to 70
  Repair adds "Additional Experience: 30" → HALLUCINATION

ROOT CAUSE: The repair prompt says "sum_check MUST equal {doc_total}" which
forces the LLM to invent marks to fill the gap, even when there's no gap in
the actual RFP (the RFP page shows 5 criteria × 20 marks max = 70, not 100).

FIX: Add a guard before calling repair:
  1. If the extracted doc_total itself seems wrong (e.g. live_marks were
     double-counted), recompute it from the criteria sum.
  2. Only call repair if delta > threshold AND the LLM explicitly flagged
     "skipped_rows" or "extraction_notes" suggesting missing content.
  3. In the repair prompt, explicitly say: "if no additional rows exist,
     return the same criteria with sum_check = {actual_sum}; do NOT invent
     criteria to reach {doc_total}."

INTEGRATION: This patch replaces _extract_with_llm() in tq_extractor.py.
Copy the patched function over the original, or import it:

    from core.tq_extractor_patch import extract_with_llm_safe
    # then in run_tq_evaluation, replace:
    #   grand_total, live_marks, live_label, threshold, criteria = _extract_with_llm(eval_text)
    # with:
    #   grand_total, live_marks, live_label, threshold, criteria = extract_with_llm_safe(eval_text)
"""

from __future__ import annotations

import json
from core.llm_client import call_llm, extract_json

# Reuse validation logic from extractor
import re

_LIVE_PATTERNS = re.compile(
    r"(presentation\b|interview\b|viva\b|panel\s+discussion|"
    r"technical\s+presentation|virtual\s+presentation)",
    re.IGNORECASE,
)
_SKIP_PATTERNS = re.compile(
    r"(^presentation$|^interview$|viva\b|^demo$|^panel$|financial\s+bid|"
    r"price\s+bid|\bL1\b|commercial\s+bid|indemnity|arbitration|"
    r"combined\s+and\s+final|appreciation\s+and\s+response|"
    r"evaluation\s+of\s+financial|opening\s+of.*financial)",
    re.IGNORECASE | re.VERBOSE,
)


def _validate_criterion(c: dict) -> bool:
    name  = (c.get("parameter") or "").strip()
    marks = c.get("max_marks", 0)
    if not name or not marks or marks < 1:
        return False
    if _SKIP_PATTERNS.search(name) or _LIVE_PATTERNS.search(name):
        return False
    return True


# ── Verbatim extraction prompt (unchanged from v18) ──────────────────────────
_EXTRACTION_PROMPT = """\
You are analysing the Technical Bid Evaluation section of a government RFP.
[PAGE N] markers indicate page breaks.

YOUR THREE TASKS:
  1. Extract grand_total marks (ALL criteria INCLUDING any live presentation)
  2. Identify live_assessment_marks (presentations/interviews scored by a panel)
  3. Extract every document-scoreable criterion with verbatim scoring rules
     and search_keywords

CRITICAL:
- Each numbered S.No row = ONE criterion.
- Lettered sub-rows with their OWN marks = sub-criteria.
- DO NOT invent criteria. Only extract what is EXPLICITLY in the table.
- If the criteria you find sum to less than (grand_total - live_marks),
  set extraction_notes to explain the discrepancy and set skipped_rows.
  NEVER fill a gap by inventing a "Additional Experience" or similar row.

RFP TEXT:
{text}

Return ONLY valid JSON:
{{
  "grand_total": <int>,
  "live_assessment_marks": <int or 0>,
  "live_assessment_label": "<label>",
  "doc_total": <grand_total - live_marks>,
  "qualification_threshold_pct": <number or null>,
  "criteria": [
    {{
      "item_code": "1",
      "parameter": "<short name <60 chars>",
      "max_marks": <int>,
      "formula_type": "BAND|BINARY|STEP|QUAL|LLM",
      "criteria_text": "<VERBATIM scoring rules>",
      "search_keywords": ["<kw1>","<kw2>"],
      "sub_criteria": []
    }}
  ],
  "sum_check": <sum of top-level max_marks>,
  "skipped_rows": [],
  "extraction_notes": "<any issues or gaps — be honest>"
}}
"""

# ── Patched repair prompt — explicitly prohibits invention ────────────────────
_REPAIR_PROMPT_SAFE = """\
You previously extracted technical evaluation criteria from an RFP.
The criteria sum to {actual_sum} marks, but you reported doc_total={doc_total}.

IMPORTANT — Before adding anything:
  • Re-read the RFP text below carefully.
  • If there genuinely ARE additional rows you missed, add them now.
  • If there are NO additional rows, return the SAME criteria list with
    sum_check = {actual_sum}.  DO NOT invent criteria to reach {doc_total}.
    Instead, correct doc_total to match {actual_sum}.

Common real causes of a gap:
  - Lettered sub-rows (a., b.) not extracted as sub_criteria
  - A row split across pages was missed

Common WRONG causes (do NOT use these):
  - Inventing a vague criterion like "Additional Experience" or
    "General Consulting Experience" to fill the mark gap

Previous extraction:
{previous_json}

RFP text (pages most likely to contain missed rows):
{text}

Return ONLY corrected, complete JSON.
If no additional criteria exist, set doc_total = sum_check = {actual_sum}.
"""


def extract_with_llm_safe(eval_text: str) -> tuple[int, int, str, float, list]:
    """
    Safe replacement for tq_extractor._extract_with_llm().
    Adds hallucination guard: repair pass cannot invent criteria.

    Returns (grand_total, live_marks, live_label, threshold, criteria).
    """
    prompt = _EXTRACTION_PROMPT.format(text=eval_text)
    print(f"[TQ-safe] Extraction prompt: {len(prompt):,} chars → LLM")

    raw    = call_llm(prompt, label="tq-extract-safe")
    parsed = extract_json(raw)

    if not parsed:
        print("[TQ-safe] LLM returned no parseable JSON")
        return 100, 0, "", 70.0, []

    grand_total = int(parsed.get("grand_total") or 100)
    live_marks  = int(parsed.get("live_assessment_marks") or 0)
    live_label  = str(parsed.get("live_assessment_label") or "")
    doc_total   = int(parsed.get("doc_total") or (grand_total - live_marks))
    threshold   = float(parsed.get("qualification_threshold_pct") or 70.0)

    if live_marks:
        print(f"[TQ-safe] Live assessment: {live_marks} marks ({live_label})")

    raw_criteria = parsed.get("criteria", [])
    criteria     = [c for c in raw_criteria if _validate_criterion(c)]
    actual_sum   = sum(c.get("max_marks", 0) for c in criteria)

    print(f"[TQ-safe] Extraction: grand={grand_total}, live={live_marks}, "
          f"doc_total={doc_total}, criteria={len(criteria)}, sum={actual_sum}")

    delta = abs(actual_sum - doc_total)

    if delta > 2:
        # ── GUARD: only repair if LLM flagged skipped rows ──────────────────
        skipped       = parsed.get("skipped_rows", [])
        notes         = parsed.get("extraction_notes", "")
        has_real_gaps = bool(skipped) or "miss" in notes.lower() or "split" in notes.lower()

        if not has_real_gaps:
            print(f"[TQ-safe] Sum mismatch ({actual_sum} vs {doc_total}) but no skipped "
                  f"rows flagged → accepting actual_sum={actual_sum} as correct doc_total")
            # Correct doc_total rather than hallucinating criteria
            doc_total = actual_sum
        else:
            print(f"[TQ-safe] Sum mismatch + skipped rows flagged → safe repair pass")
            rp = _REPAIR_PROMPT_SAFE.format(
                doc_total    = doc_total,
                actual_sum   = actual_sum,
                previous_json= json.dumps(parsed, indent=2)[:4000],
                text         = eval_text[:5000],
            )
            raw2    = call_llm(rp, label="tq-repair-safe")
            parsed2 = extract_json(raw2)
            if parsed2 and parsed2.get("criteria"):
                crit2   = [c for c in parsed2["criteria"] if _validate_criterion(c)]
                actual2 = sum(c.get("max_marks", 0) for c in crit2)
                print(f"[TQ-safe] After repair: {len(crit2)} criteria, sum={actual2}")

                # Only accept if repair improved the match AND didn't just add vague criteria
                vague_names = {"additional experience", "general consulting",
                               "other experience", "additional marks"}
                new_names = {c.get("parameter","").lower() for c in crit2
                             if c not in criteria}
                if any(v in n for n in new_names for v in vague_names):
                    print("[TQ-safe] Repair invented vague criteria — REJECTED, using original")
                elif abs(actual2 - doc_total) < delta:
                    criteria  = crit2
                    doc_total = actual2

    return grand_total, live_marks, live_label, threshold, criteria
