"""
tq_extractor_patch.py — v17 delta
==================================

Apply these changes on top of tq_extractor v16.

FIXES
-----
FIX 1  int('3(a)') crash in criteria.sort()
       _safe_sort_key() converts any item_code to a comparable tuple:
         "1"    → (1, "")
         "3"    → (3, "")
         "3(a)" → (3, "a")
         "3a"   → (3, "a")
         "10"   → (10, "")

FIX 2  Comparative / proportionate scoring detection
       Some RFPs (QCBS/World Bank style) score on a RELATIVE basis:
         "the Applicant that has undertaken the highest number of Eligible
          Assignments shall be entitled to the maximum score... all other
          shall be entitled to a proportionate score."
       This CANNOT be scored from a single proposal — you need all bidders.
       These criteria are tagged requires_comparative_evaluation=True and
       scored as "-2" sentinel so the UI shows "Comparative — pending".

FIX 3  Hierarchical item codes (3(a), 3(b) … are sub-items of 3)
       When the parent item_code "3" exists in the criteria list AND sub-items
       "3(a)", "3(b)" etc. also exist, the parent row is promoted to a
       summary row (is_sub_item=False, max_marks = sum of children) and
       the children are tagged is_sub_item=True with parent_parameter set.
       This preserves the full mark structure without double-counting.

FIX 4  grand_total sort: use _safe_sort_key() everywhere int() was used.

HOW TO APPLY
------------
In tq_extractor.py, replace / add the functions below.
Search for the affected function names and swap them out.
The run_tq_evaluation() function needs two additional lines (marked ADD).
"""

from __future__ import annotations

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 + 4 — Safe item code sort key
# ─────────────────────────────────────────────────────────────────────────────

def _safe_sort_key(item_code: str) -> tuple:
    """
    Convert any item_code string to a sortable (int, str) tuple.

    "1"    → (1,  "")
    "3"    → (3,  "")
    "3(a)" → (3,  "a")
    "3a"   → (3,  "a")
    "3(b)" → (3,  "b")
    "10"   → (10, "")
    "99"   → (99, "")   ← fallback
    """
    code = str(item_code or "99").strip()

    # Match "3(a)", "3(b)", "3 (a)"
    m = re.match(r"^(\d+)\s*[\(\[]\s*([a-zA-Z])\s*[\)\]]$", code)
    if m:
        return (int(m.group(1)), m.group(2).lower())

    # Match "3a", "3b"
    m = re.match(r"^(\d+)([a-zA-Z])$", code)
    if m:
        return (int(m.group(1)), m.group(2).lower())

    # Pure integer "1", "10", "3"
    m = re.match(r"^(\d+)$", code)
    if m:
        return (int(m.group(1)), "")

    return (99, code.lower())


def _sort_criteria(criteria: list) -> list:
    """Sort criteria by _safe_sort_key on item_code. Replaces raw int() sort."""
    return sorted(criteria, key=lambda x: _safe_sort_key(str(x.get("item_code", "99"))))


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Comparative scoring detection
# ─────────────────────────────────────────────────────────────────────────────

_COMPARATIVE_SCORING = re.compile(
    r"""(
        proportionate\s+score                    |
        highest\s+number.*?maximum\s+score       |
        maximum\s+score.*?proportionate          |
        relative\s+scoring                       |
        comparative\s+(size|quality|evaluation)  |
        ranked\s+from\s+highest\s+to\s+lowest    |
        qcbs\b                                   |
        quality.*?cost.*?based\s+selection       |
        sf\s*=\s*100\s*[×x\*]\s*fm               |
        combined\s+(technical\s+and\s+financial\s+)?score
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_COMPARATIVE_PARAM = re.compile(
    r"""(
        methodology\s+and\s+work\s+plan          |
        proposed\s+methodology                   |
        work\s+plan
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _is_comparative(parameter: str, criteria_text: str) -> bool:
    """
    Return True when this criterion uses proportionate/comparative scoring
    that requires all proposals to be evaluated together.

    Two signals:
    A. The criteria_text itself mentions proportionate/comparative scoring.
    B. The parameter is "Methodology/Work Plan" AND the overall RFP uses QCBS.

    We detect signal A per-criterion.
    Signal B is handled in _tag_comparative_criteria() at the table level.
    """
    combined = f"{criteria_text}"
    return bool(_COMPARATIVE_SCORING.search(combined))


def _tag_comparative_criteria(criteria: list, table_criteria_text: str = "") -> list:
    """
    Tag criteria that cannot be scored from a single document.

    Also detects table-level signals (QCBS, combined score formula) and
    marks methodology/work-plan rows as comparative when the whole RFP is
    QCBS-style.
    """
    # Is this a QCBS / comparative-scoring RFP overall?
    qcbs_rfp = bool(_COMPARATIVE_SCORING.search(table_criteria_text))

    tagged = []
    for c in criteria:
        is_comp = (
            _is_comparative(c.get("parameter", ""), c.get("criteria_text", ""))
            or (qcbs_rfp and bool(_COMPARATIVE_PARAM.search(c.get("parameter", ""))))
        )
        tagged.append({**c, "_comparative": is_comp})

    return tagged


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Hierarchical item code resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_hierarchy(criteria: list) -> list:
    """
    When sub-items (3(a), 3(b) …) exist alongside their parent (3):

      • Sub-items → is_sub_item=True, parent_parameter = parent["parameter"]
      • Parent    → max_marks updated to sum of children (if children's sum
                    is closer to stated marks than the cell value alone)

    When no parent row exists but sub-items do (parent was dropped by Docling):
      • A synthetic parent row is created from the sub-items' common stem.

    When only one level exists (no sub-items), nothing changes.
    """
    by_key: dict[tuple, dict] = {}
    for c in criteria:
        key = _safe_sort_key(str(c.get("item_code", "99")))
        by_key[key] = c

    # Find all integer-only parents
    parent_keys   = {k for k in by_key if k[1] == ""}
    children_keys = {k for k in by_key if k[1] != ""}

    if not children_keys:
        return criteria   # no hierarchical codes at all

    # Group children by their parent numeric index
    from collections import defaultdict
    child_groups: dict[int, list] = defaultdict(list)
    for k in children_keys:
        child_groups[k[0]].append(k)

    result = []
    processed_parents: set = set()

    for parent_num, child_ks in child_groups.items():
        parent_key = (parent_num, "")
        children   = sorted([by_key[k] for k in child_ks],
                            key=lambda c: _safe_sort_key(str(c.get("item_code", ""))))
        child_sum  = sum(c.get("max_marks", 0) for c in children)

        if parent_key in by_key:
            parent = dict(by_key[parent_key])
            # If children's sum matches what we expect, use it
            if child_sum > parent.get("max_marks", 0):
                parent["max_marks"] = child_sum
            parent["is_sub_item"]      = False
            parent["parent_parameter"] = ""
            result.append(parent)
            processed_parents.add(parent_key)
        else:
            # Synthesise a parent from children's common parameter stem
            first_param = children[0].get("parameter", "")
            # Try to find a stem: "Project Director & Team Leader" → "Key Personnel"
            synthetic_param = _common_param_stem(first_param, children)
            result.append({
                "item_code":        str(parent_num),
                "parameter":        synthetic_param,
                "max_marks":        child_sum,
                "criteria_text":    "",
                "is_sub_item":      False,
                "parent_parameter": "",
                "_synthetic":       True,
            })

        for child in children:
            result.append({
                **child,
                "is_sub_item":      True,
                "parent_parameter": (by_key.get(parent_key, {}) or {}).get(
                    "parameter", f"Criterion {parent_num}"),
            })

    # Add non-child, non-parent-of-children rows unchanged
    for k, c in by_key.items():
        if k[1] == "" and k not in processed_parents:
            # Only add if not the parent of any child group
            if k[0] not in child_groups:
                result.append(c)

    return _sort_criteria(result)


def _common_param_stem(first_param: str, children: list) -> str:
    """
    Heuristic: derive a human-readable parent name from children list.
    E.g. children are "Project Director", "Data Lead", "Usability Lead"
    → return "Key Personnel"
    """
    params = [c.get("parameter", "") for c in children]
    # Check for common suffixes/themes
    if any("personnel" in p.lower() or "lead" in p.lower() or "director" in p.lower()
           for p in params):
        return "Relevant Experience of Key Personnel"
    if any("project" in p.lower() for p in params):
        return "Project Criteria"
    return first_param.split("&")[0].strip() or "Criterion Group"


# ─────────────────────────────────────────────────────────────────────────────
# Updated extract_marking_table() — drop-in replacement for the tail section
# (the part after _dedup_criteria)
# ─────────────────────────────────────────────────────────────────────────────
# In your existing extract_marking_table(), replace the block from
# "criteria = _dedup_criteria(criteria)" to the end of the function with
# the code below.  Also pass full_table_text into _tag_comparative_criteria.

EXTRACT_TAIL_REPLACEMENT = '''
    criteria = _dedup_criteria(criteria)

    # FIX 1/4: safe sort (no more int('3(a)') crash)
    criteria = _sort_criteria(criteria)

    # FIX 3: resolve hierarchical item codes (3(a), 3(b) → sub-items of 3)
    criteria = _resolve_hierarchy(criteria)

    # FIX 2: tag comparative criteria (QCBS / proportionate scoring)
    # Pass the raw LLM extraction context so table-level signals are detected
    raw_table_text = " ".join(c.get("criteria_text", "") for c in criteria)
    criteria = _tag_comparative_criteria(criteria, raw_table_text)

    # Drop presentation / live-assessment rows (unchanged from v16)
    criteria = [c for c in criteria
                if not _SKIP_ROW_PATTERNS.search(_normalize_param(c.get("parameter", "")))]

    doc_max     = sum(c["max_marks"] for c in criteria if not c.get("is_sub_item"))
    grand_total = meta.get("grand_total_marks") or doc_max
    threshold   = meta.get("qualification_threshold_pct") or 70.0

    schema_warning = None
    if doc_max < 20:
        schema_warning = f"Only {doc_max} marks extracted — likely missed rows."
    elif grand_total > 0 and abs(doc_max - grand_total) > 10:
        schema_warning = (f"doc_max={doc_max} but RFP declares {grand_total} — "
                          "verify manually.")

    print(f"[TQ] Final: {len(criteria)} criteria | doc_max={doc_max} | grand_total={grand_total}")
    for c in criteria:
        comp_tag = " [COMPARATIVE]" if c.get("_comparative") else ""
        sub_tag  = " [sub]"        if c.get("is_sub_item")   else ""
        print(f"  [{str(c.get("item_code","?")):5s}] "
              f"{c["parameter"][:50]:50s}  {c["max_marks"]:3d} marks{comp_tag}{sub_tag}")

    return {
        "evaluation_title":            meta.get("evaluation_title", "Technical Evaluation"),
        "grand_total_marks":           grand_total,
        "qualification_threshold_pct": threshold,
        "criteria":                    criteria,
        "doc_max":                     doc_max,
        "schema_warning":              schema_warning,
        "context_source":              f"cluster p{cluster[0] if cluster else "?"}",
        "error":                       None,
    }
'''


# ─────────────────────────────────────────────────────────────────────────────
# Updated score_criterion() — handle comparative criteria
# ─────────────────────────────────────────────────────────────────────────────
# Add this check AT THE TOP of score_criterion(), after the _is_live_assessment check:

SCORE_CRITERION_COMPARATIVE_PATCH = '''
    # FIX 2: comparative criteria cannot be scored from a single proposal
    if criterion.get("_comparative"):
        return {
            "score":                           None,   # → "-2" sentinel in DB
            "extracted_value":                 "Requires all proposals for comparative scoring",
            "source_page":                     None,
            "scoring_steps":                   "Comparative/proportionate scoring — all bidders needed",
            "justification":                   "This criterion uses proportionate scoring relative to other bidders.",
            "strengths":                       [],
            "gaps":                            ["Comparative scoring requires all submitted proposals"],
            "evidence_found":                  False,
            "requires_comparative_evaluation": True,
        }
'''


# ─────────────────────────────────────────────────────────────────────────────
# Updated run_tq_evaluation() — handle comparative scores in totals
# ─────────────────────────────────────────────────────────────────────────────
# In run_tq_evaluation(), replace the total_scored computation block:

RUN_TQ_TOTALS_PATCH = '''
    _prog("Computing totals", 96)

    # FIX 2: exclude comparative scores from the document total
    doc_scores  = [s for s in scores if not s.get("requires_comparative_evaluation")
                                     and not s.get("is_sub_item")]
    comp_scores = [s for s in scores if s.get("requires_comparative_evaluation")]

    # doc_max should also exclude sub-items (parent already has their sum)
    doc_max_actual = sum(s["max_marks"] for s in doc_scores)

    total_scored = round(sum(s.get("score") or 0 for s in doc_scores), 1)
    total_pct    = round((total_scored / doc_max_actual) * 100, 1) if doc_max_actual > 0 else 0.0
    comp_marks   = sum(s["max_marks"] for s in comp_scores)
'''


# ─────────────────────────────────────────────────────────────────────────────
# Minimal self-contained test for the three fixes
# ─────────────────────────────────────────────────────────────────────────────

def _test_fixes():
    print("=== FIX 1: _safe_sort_key ===")
    cases = ["1", "2", "3", "3(a)", "3(b)", "3(c )", "3(d)", "3(e)", "3(f)", "10", "99", "3a"]
    for c in cases:
        print(f"  {c!r:12s} → {_safe_sort_key(c)}")
    items = [{"item_code": c} for c in cases]
    sorted_items = _sort_criteria(items)
    print("  Sorted:", [x["item_code"] for x in sorted_items])

    print("\n=== FIX 2: _is_comparative ===")
    comparative_text = (
        "30% of the maximum marks shall be awarded for the number of Eligible Assignments. "
        "The remaining 70% shall be awarded for comparative size, quality and complexity. "
        "the Applicant that has undertaken the highest number shall be entitled to the "
        "maximum score and all other competing Applicants shall be entitled to a proportionate score."
    )
    non_comparative = "Single order of 6 professionals: 10 marks. More than 12: 20 marks."
    print(f"  comparative_text → {_is_comparative('Relevant Experience', comparative_text)}")
    print(f"  non_comparative  → {_is_comparative('Past Experience A', non_comparative)}")

    print("\n=== FIX 3: _resolve_hierarchy ===")
    sample_criteria = [
        {"item_code": "1",    "parameter": "Relevant Experience of Applicant", "max_marks": 30, "criteria_text": ""},
        {"item_code": "2",    "parameter": "Proposed Methodology and Work Plan", "max_marks": 10, "criteria_text": ""},
        {"item_code": "3",    "parameter": "Relevant Experience of Key Personnel", "max_marks": 60, "criteria_text": ""},
        {"item_code": "3(a)", "parameter": "Project Director & Team Leader", "max_marks": 20, "criteria_text": ""},
        {"item_code": "3(b)", "parameter": "Data Oversight and Quality: Lead", "max_marks": 10, "criteria_text": ""},
        {"item_code": "3(c)", "parameter": "Project Management Team: Lead", "max_marks": 10, "criteria_text": ""},
        {"item_code": "3(d)", "parameter": "User Engagement Team: Lead", "max_marks": 10, "criteria_text": ""},
        {"item_code": "3(e)", "parameter": "Usability Lead", "max_marks": 5, "criteria_text": ""},
        {"item_code": "3(f)", "parameter": "Technology Tools Lead", "max_marks": 5, "criteria_text": ""},
    ]
    resolved = _resolve_hierarchy(sample_criteria)
    for c in resolved:
        sub = " [sub]" if c.get("is_sub_item") else ""
        print(f"  [{c['item_code']:5s}] {c['parameter'][:45]:45s} {c['max_marks']:3d}{sub}")

    top_level_marks = sum(c["max_marks"] for c in resolved if not c.get("is_sub_item"))
    print(f"\n  Top-level doc_max (no double-counting): {top_level_marks}")
    assert top_level_marks == 100, f"Expected 100, got {top_level_marks}"
    print("  ✓ Grand total correct")


if __name__ == "__main__":
    _test_fixes()
