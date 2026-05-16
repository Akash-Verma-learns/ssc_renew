"""
core/tq_extractor.py  (REPLACEMENT — drop-in for the existing file)
=====================================================================
Main orchestrator for the TQ (Technical Qualification) evaluation pipeline.

Preserves the exact same public interface as the original so routes.py
needs ZERO changes:

    from core.tq_extractor import run_tq_evaluation, ingest_proposal

Pipeline
--------
Stage 1  extract_marking_scheme()    core/tq_criteria_extractor.py
Stage 2  score_criterion()           core/tq_proposal_scorer.py
Stage 3  Aggregate + discrepancy     here
Stage 4  Qualification gate          here

All Ollama calls use local models only. No API keys required.
Graceful degradation: any stage failure is caught; partial results saved.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ── Lazy imports (avoid circular on routes.py startup) ───────────────────────

def _criteria_extractor():
    from core.tq_criteria_extractor import extract_marking_scheme
    return extract_marking_scheme

def _scorer():
    from core.tq_proposal_scorer import score_criterion
    return score_criterion

def _parser():
    from core.parser import parse_document
    return parse_document

def _ingest():
    from core.vector_store import ingest_chunks
    return ingest_chunks


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TQ_UPLOAD_DIR  = Path("./tq_uploads")
RFP_UPLOAD_DIR = Path("./uploads")
TQ_UPLOAD_DIR.mkdir(exist_ok=True)

_DB_PARAM_MAX = 295   # TQScoreItem.parameter column length limit


def _trunc(text: str, n: int = _DB_PARAM_MAX) -> str:
    if text and len(text) > n:
        return text[:n - 3] + "..."
    return text or ""


# ──────────────────────────────────────────────────────────────────────────────
# Proposal ingestion (unchanged interface)
# ──────────────────────────────────────────────────────────────────────────────

def ingest_proposal(proposal_path: str, proposal_doc_name: str) -> int:
    """Parse the proposal PDF/DOCX and ingest into ChromaDB."""
    print(f"[TQ] Ingesting proposal: {Path(proposal_path).name}")
    parse_fn  = _parser()
    ingest_fn = _ingest()
    chunks    = parse_fn(proposal_path)
    for c in chunks:
        c.doc_name = proposal_doc_name
        c.chunk_id = f"{proposal_doc_name}_{c.page_no}_{c.chunk_id.split('_')[-1]}"
    count = ingest_fn(chunks, doc_id=proposal_doc_name)
    print(f"[TQ] Proposal ingested: {count} chunks")
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Discrepancy aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_discrepancies(scores: list[dict]) -> list[str]:
    """Collect and deduplicate discrepancies across all scored criteria."""
    seen: set[str] = set()
    out:  list[str] = []
    for s in scores:
        for d in (s.get("discrepancies") or []):
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


def _fy_pattern_summary(scores: list[dict]) -> Optional[str]:
    """
    Detect whether the proposal consistently uses wrong financial years.
    If ≥2 criteria have the same year mismatch, surface it as a global flag.
    """
    fy_discs = [d for s in scores for d in (s.get("discrepancies") or [])
                if "FINANCIAL YEAR" in d]
    if len(fy_discs) >= 2:
        return fy_discs[0]   # representative message
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Qualification gate
# ──────────────────────────────────────────────────────────────────────────────

def _qualification_gate(
    total_scored: float,
    doc_max: int,
    threshold_pct: Optional[float],
    global_discrepancies: list[str],
) -> dict:
    if not threshold_pct or doc_max == 0:
        return {}

    achieved_pct = round((total_scored / doc_max) * 100, 1)
    passed       = achieved_pct >= threshold_pct
    disc_note    = (
        f" Note: {len(global_discrepancies)} discrepancy/discrepancies flagged "
        f"(see discrepancies list)."
        if global_discrepancies else ""
    )

    return {
        "threshold_pct":       float(threshold_pct),
        "achieved_pct":        achieved_pct,
        "passed":              passed,
        "financial_bid_opens": passed,
        "note": (
            f"{'QUALIFIED' if passed else 'NOT QUALIFIED'} — "
            f"{achieved_pct}% vs ≥{threshold_pct}% required.{disc_note}"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def _load_old_rfp_cache(rfp_path: Path) -> Optional[dict]:
    """
    Try to read from the old v19 rfp_cache/ directory.
    Returns the raw cache dict or None.
    """
    try:
        from core.rfp_cache import load_cache as _load_cache
        return _load_cache(str(rfp_path))
    except Exception as e:
        print(f"[TQ] rfp_cache lookup skipped: {e}")
        return None


def _old_cache_to_table(cached: dict) -> dict:
    """
    Convert old rfp_cache format → the 'table' dict shape that
    run_tq_evaluation() expects internally.

    Old cache keys:  grand_total, live_marks, live_label, doc_total,
                     threshold, criteria (flat list), bands
    New table keys:  grand_total_marks, live_assessment_marks,
                     live_assessment_label, doc_max,
                     qualification_threshold_pct, criteria, rfp_bands
    """
    grand_total = int(cached.get("grand_total") or 100)
    live_marks  = int(cached.get("live_marks")  or 0)
    live_label  = str(cached.get("live_label")  or "")
    doc_total   = int(cached.get("doc_total")   or (grand_total - live_marks))
    threshold   = float(cached.get("threshold") or 70.0)
    rfp_bands   = cached.get("bands", {})

    raw_criteria = cached.get("criteria", [])

    # Convert each criterion:
    #   formula_type  → formula_hint   (old key → new key)
    #   inject _cached_bands for the formula engine
    criteria_out = []
    for c in raw_criteria:
        param = c.get("parameter", "")
        new_c = dict(c)
        if "formula_hint" not in new_c:
            new_c["formula_hint"] = c.get("formula_type", "LLM")
        band_data = rfp_bands.get(param)
        if band_data:
            new_c["_cached_bands"] = band_data
        criteria_out.append(new_c)

    return {
        "criteria":                    criteria_out,
        "grand_total_marks":           grand_total,
        "live_assessment_marks":       live_marks,
        "live_assessment_label":       live_label,
        "doc_max":                     doc_total,
        "qualification_threshold_pct": threshold,
        "schema_warning":              None,
        "error":                       None,
        "_from_cache":                 True,
    }


def run_tq_evaluation(
    rfp_doc_name:      str,
    proposal_path:     str,
    proposal_doc_name: str,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Full TQ evaluation pipeline.

    Parameters
    ----------
    rfp_doc_name      : filename in ./uploads/ (e.g. "abc12345.pdf")
    proposal_path     : absolute path to the proposal PDF/DOCX
    proposal_doc_name : unique identifier for the proposal in ChromaDB
    progress_callback : optional callable(step: str, pct: int)

    Returns standard result dict consumed by routes.py / run_tq_evaluation_task().

    Stage 1 — Criteria extraction (priority order):
        1. Old rfp_cache/ (v19 cache) — deterministic, correct if already run
        2. New waterfall extractor    — TOC → page-score → text-line → LLM
    """

    def _prog(step: str, pct: int):
        if progress_callback:
            try: progress_callback(step, pct)
            except Exception: pass
        print(f"[TQ] {pct:3d}%  {step}")

    # ── Resolve RFP file path ─────────────────────────────────────────────────
    rfp_path = RFP_UPLOAD_DIR / rfp_doc_name
    if not rfp_path.exists():
        candidates = list(RFP_UPLOAD_DIR.glob(f"{rfp_doc_name.split('.')[0]}*"))
        if candidates:
            rfp_path = candidates[0]
        else:
            err = f"RFP file not found: {rfp_doc_name}"
            print(f"[TQ] ERROR: {err}")
            return _empty_result(error=err)

    # ── Stage 1: Criteria extraction ──────────────────────────────────────────
    _prog("Reading RFP marking scheme", 5)

    table = None

    # 1a. Try old rfp_cache first (most reliable — already validated by v19)
    old_cached = _load_old_rfp_cache(rfp_path)
    if old_cached and old_cached.get("criteria"):
        table = _old_cache_to_table(old_cached)
        n_crit = len([c for c in table["criteria"] if not c.get("is_parent")])
        grand  = table["grand_total_marks"]
        live   = table["live_assessment_marks"]
        print(f"[TQ] ✓ rfp_cache hit: {n_crit} scoreable criteria, "
              f"grand={grand}, live={live}, doc={table['doc_max']}")

    # 1b. Fall back to new waterfall extractor
    if table is None:
        extract_fn = _criteria_extractor()
        try:
            table = extract_fn(str(rfp_path))
        except Exception as e:
            table = {"criteria": [], "error": str(e), "grand_total_marks": 0,
                     "live_assessment_marks": 0, "live_assessment_label": "",
                     "qualification_threshold_pct": 70.0, "schema_warning": None}
        print(f"[TQ] New waterfall extraction: "
              f"{len(table.get('criteria', []))} criteria, "
              f"grand={table.get('grand_total_marks', 0)}")

    all_criteria = table.get("criteria", [])
    if not all_criteria:
        _prog("Failed: no criteria extracted from RFP", 100)
        return _empty_result(
            error=table.get("error") or "No criteria extracted from RFP",
            schema_warning=table.get("schema_warning"),
        )

    # Separate scoreable (non-parent) from display-only parents
    scoreable_criteria = [c for c in all_criteria if not c.get("is_parent")]
    parent_criteria    = [c for c in all_criteria if c.get("is_parent")]

    # Use pre-computed doc_max from cache; fall back to sum of scoreable marks
    doc_max   = int(table.get("doc_max") or
                    sum(c["max_marks"] for c in scoreable_criteria))
    threshold = table.get("qualification_threshold_pct", 70.0)
    warn      = table.get("schema_warning")
    live_marks = int(table.get("live_assessment_marks") or 0)
    live_label = str(table.get("live_assessment_label") or "")
    grand_total = int(table.get("grand_total_marks") or (doc_max + live_marks))

    _prog(f"Found {len(scoreable_criteria)} scoreable criteria "
          f"({doc_max} doc marks, {live_marks} live)", 12)

    # ── Stage 2: Ingest proposal ──────────────────────────────────────────────
    _prog("Ingesting proposal into vector store", 15)
    try:
        ingest_proposal(proposal_path, proposal_doc_name)
    except Exception as e:
        print(f"[TQ] Proposal ingest warning: {e} — continuing without vector store")

    # ── Stage 3: Score each criterion ─────────────────────────────────────────
    _prog("Scoring criteria against proposal", 20)
    score_fn = _scorer()
    scores: list[dict] = []
    n = len(scoreable_criteria)

    for i, criterion in enumerate(scoreable_criteria):
        pct  = 20 + int((i / max(n, 1)) * 68)
        name = criterion.get("parameter", "")[:55]
        _prog(f"Scoring: {name}", pct)

        try:
            result = score_fn(criterion, proposal_path)
        except Exception as e:
            print(f"[TQ] Scoring error for '{name}': {e}")
            result = {
                "score": 0, "score_percentage": 0.0,
                "extracted_value": None, "source_page": None,
                "raw_evidence": None, "pages_searched": [],
                "scoring_steps": f"Error: {e}",
                "justification": f"Scoring failed: {e}",
                "discrepancies": [], "strengths": [], "gaps": [str(e)],
                "evidence_found": False,
            }

        sc  = result.get("score", 0)
        pg  = f"(p.{result['source_page']})" if result.get("source_page") else ""
        mm  = criterion["max_marks"]
        dis = " ⚠" if result.get("discrepancies") else ""
        print(f"  [{i+1:2d}/{n}] {name:55s}  {sc:5.1f}/{mm}{dis} {pg}")

        scores.append({
            "item_code":       _trunc(criterion.get("item_code", str(i + 1)), 20),
            "parameter":       _trunc(name, _DB_PARAM_MAX),
            "max_marks":       mm,
            "criteria_text":   criterion.get("criteria_text", ""),
            "formula_hint":    criterion.get("formula_hint", criterion.get("formula_type", "LLM")),
            "is_sub_item":     bool(criterion.get("is_sub_item", False)),
            "parent_parameter": criterion.get("parent_parameter", ""),
            **result,
            "evaluation_layer":                "document",
            "requires_live_assessment":        False,
            "requires_comparative_evaluation": False,
        })

    # ── Parent summary rows (display only — sum of children) ─────────────────
    for p in parent_criteria:
        p_param = p.get("parameter", "")
        children = [s for s in scores if s.get("parent_parameter") == p_param]
        p_score  = round(sum(c.get("score") or 0 for c in children), 1)
        scores.append({
            "item_code":       _trunc(p.get("item_code", ""), 20),
            "parameter":       _trunc(p_param, _DB_PARAM_MAX),
            "max_marks":       p["max_marks"],
            "criteria_text":   p.get("criteria_text", ""),
            "formula_hint":    p.get("formula_hint", p.get("formula_type", "QUAL")),
            "is_sub_item":     False,
            "parent_parameter": "",
            "score":           p_score,
            "score_percentage": round((p_score / p["max_marks"]) * 100, 1) if p["max_marks"] else 0,
            "extracted_value": "Sum of sub-criteria",
            "source_page":     None,
            "scoring_steps":   f"Parent = sum of sub-criteria = {p_score}",
            "justification":   f"Score {p_score}/{p['max_marks']} (sum of sub-criteria)",
            "discrepancies":   [],
            "strengths":       [],
            "gaps":            [],
            "evidence_found":  p_score > 0,
            "evaluation_layer": "document",
            "requires_live_assessment":        False,
            "requires_comparative_evaluation": False,
        })

    # ── Live assessment placeholder ───────────────────────────────────────────
    if live_marks > 0:
        scores.append({
            "item_code":       "L1",
            "parameter":       _trunc(live_label or "Technical Presentation", _DB_PARAM_MAX),
            "max_marks":       live_marks,
            "criteria_text":   "Live panel presentation — scored by committee",
            "formula_hint":    "LIVE",
            "is_sub_item":     False,
            "parent_parameter": "",
            "score":           None,
            "score_percentage": None,
            "extracted_value": "Pending panel evaluation",
            "source_page":     None,
            "scoring_steps":   "Live assessment — cannot score from document",
            "justification":   "Pending live presentation evaluation",
            "discrepancies":   [],
            "strengths":       [],
            "gaps":            ["Live panel evaluation required"],
            "evidence_found":  False,
            "evaluation_layer":                "live_assessment",
            "requires_live_assessment":        True,
            "requires_comparative_evaluation": False,
        })

    # ── Stage 4: Aggregate ────────────────────────────────────────────────────
    _prog("Computing totals and qualification gate", 93)

    # Total scored = leaf (non-parent, non-live) scores only
    leaf_scores  = [s for s in scores
                    if s.get("evaluation_layer") == "document"
                    and s.get("extracted_value") != "Sum of sub-criteria"
                    and not s.get("requires_live_assessment")]
    total_scored = round(sum(s.get("score") or 0 for s in leaf_scores), 1)
    total_pct    = round((total_scored / doc_max) * 100, 1) if doc_max > 0 else 0.0

    global_discs = _aggregate_discrepancies(scores)
    fy_flag      = _fy_pattern_summary(scores)
    qualification = _qualification_gate(total_scored, doc_max, threshold, global_discs)

    if fy_flag and fy_flag not in global_discs:
        global_discs.insert(0, f"GLOBAL: {fy_flag}")

    _prog("Done", 100)

    from_cache_msg = " [rfp_cache]" if table.get("_from_cache") else " [fresh extract]"
    print(f"\n[TQ] ── Result{from_cache_msg} ────────────────────────────────")
    print(f"[TQ] Grand: {grand_total} | Doc: {doc_max} | Live: {live_marks}")
    print(f"[TQ] Scored: {total_scored} / {doc_max}  ({total_pct}%)")
    if qualification:
        verdict = "QUALIFIED ✅" if qualification.get("passed") else "NOT QUALIFIED ❌"
        print(f"[TQ] Gate ({threshold}%): {verdict}")
    if global_discs:
        print(f"[TQ] ⚠ {len(global_discs)} discrepancy/ies flagged")
    if warn:
        print(f"[TQ] Schema warning: {warn}")
    print(f"[TQ] ──────────────────────────────────────────────────────\n")

    return {
        "evaluation_title":            "Technical Evaluation",
        "grand_total_marks":           grand_total,
        "technical_document_max":      doc_max,
        "scoreable_total":             doc_max,
        "live_assessment_marks":       live_marks,
        "live_assessment_label":       live_label,
        "financial_marks":             0,
        "total_scored":                total_scored,
        "total_percentage":            total_pct,
        "final_score_formula":         None,
        "qualification_threshold":     threshold,
        "qualification":               qualification,
        "schema_valid":                warn is None,
        "schema_warning":              warn,
        "global_discrepancies":        global_discs,
        "criteria_structure":          all_criteria,
        "scores":                      scores,
        "error":                       table.get("error"),
        "from_cache":                  bool(table.get("_from_cache")),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Empty result (error cases)
# ──────────────────────────────────────────────────────────────────────────────

def _empty_result(error: str = "", schema_warning: str = None) -> dict:
    return {
        "evaluation_title":       "Technical Evaluation",
        "grand_total_marks":      0,
        "technical_document_max": 0,
        "scoreable_total":        0,
        "live_assessment_marks":  0,
        "financial_marks":        0,
        "total_scored":           0,
        "total_percentage":       0.0,
        "final_score_formula":    None,
        "qualification_threshold": 70.0,
        "qualification":          {},
        "schema_valid":           False,
        "schema_warning":         schema_warning,
        "global_discrepancies":   [],
        "criteria_structure":     [],
        "scores":                 [],
        "error":                  error,
    }