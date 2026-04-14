"""
core/gap_analyzer.py — v1  Actionable Gap Analysis
====================================================

PURPOSE
───────
After scoring, the evaluator knows WHERE marks were lost but not WHY or HOW
to fix it.  This module takes the scored results and generates:

  1. Per-criterion gap cards  — what's missing, what evidence would fix it
  2. A prioritised action list — ranked by "marks at stake × ease of fix"
  3. A qualification risk assessment — how close to the 70% threshold
  4. A markdown/HTML report suitable for sending to the proposal team

ARCHITECTURE
────────────
  Input:   scores[] from run_tq_evaluation()
           criteria_structure[] from extract_marking_table()
           raw proposal path (for evidence checking)
  Stage 1: Classify each gap by type (missing_evidence / partial / unverified)
  Stage 2: LLM gap explanation — one call per missed criterion
  Stage 3: Priority ranking — sort by (marks_lost × fix_difficulty⁻¹)
  Stage 4: Report generation — markdown + JSON

USAGE
─────
    from core.gap_analyzer import analyze_gaps, render_gap_report

    analysis = analyze_gaps(scores, doc_max=100, threshold_pct=70)
    report   = render_gap_report(analysis, rfp_title="DDU-GKY Pune/Nagpur")
    print(report["markdown"])
    print(report["json"])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from core.llm_client import call_llm, extract_json


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CriterionGap:
    parameter:          str
    max_marks:          int
    achieved_marks:     float
    marks_lost:         float
    gap_type:           str   # "missing_evidence" | "partial" | "unverified" | "full"
    root_cause:         str   # one-line diagnosis
    fix_actions:        list[str]   # concrete things to add to proposal
    evidence_keywords:  list[str]   # exact strings to search for in supporting docs
    fix_difficulty:     str   # "easy" | "medium" | "hard"
    priority_score:     float  # marks_lost / fix_difficulty_weight  (higher = fix first)
    is_sub_item:        bool
    parent_parameter:   str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GapAnalysis:
    total_scored:        float
    doc_max:             int
    threshold_pct:       float
    min_required:        float
    marks_deficit:       float       # negative = above threshold
    qualification_risk:  str         # "safe" | "marginal" | "at_risk" | "failed"
    gaps:                list[CriterionGap] = field(default_factory=list)
    top_priorities:      list[CriterionGap] = field(default_factory=list)
    recoverable_marks:   float = 0.0  # max marks achievable from "easy" fixes
    summary:             str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Gap type classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_gap(score_entry: dict) -> str:
    """Classify why marks were lost."""
    achieved = score_entry.get("score") or 0
    max_m    = score_entry.get("max_marks", 1)
    ev_found = score_entry.get("evidence_found", False)
    verified = score_entry.get("verified", True)

    if not ev_found:
        return "missing_evidence"
    if not verified:
        return "unverified"
    if achieved < max_m * 0.5:
        return "partial"
    if achieved < max_m:
        return "partial"
    return "full"


_DIFFICULTY_WEIGHTS = {"easy": 1.0, "medium": 0.5, "hard": 0.25}


def _estimate_difficulty(gap_type: str, marks_lost: float,
                         parameter: str) -> str:
    """Estimate how hard it is to recover lost marks."""
    param_lower = parameter.lower()
    # Hard: experience/history you can't fabricate
    if any(kw in param_lower for kw in ["experience", "turnover", "years", "tsa", "ddu"]):
        if gap_type == "missing_evidence":
            return "hard"   # can't claim experience you don't have
        return "medium"     # have it, just need to document it better

    # Easy: formatting / explicit statement issues
    if gap_type == "unverified":
        return "easy"   # just add page references
    if gap_type == "partial" and marks_lost <= 5:
        return "easy"

    return "medium"


# ─────────────────────────────────────────────────────────────────────────────
# LLM gap explanation (one call per missed criterion)
# ─────────────────────────────────────────────────────────────────────────────

_GAP_EXPLAIN_PROMPT = """\
You are a proposal review expert helping a bidder understand why they lost marks
on a government RFP technical evaluation.

CRITERION: {parameter}
MAX MARKS: {max_marks}
ACHIEVED:  {achieved} marks
MARKS LOST: {marks_lost}
GAP TYPE:  {gap_type}
SCORING RULES: {criteria_text}
WHAT WAS FOUND IN PROPOSAL: {extracted_value}
EVALUATOR COMMENT: {justification}

Your job:
1. Diagnose the exact root cause in one sentence
2. List 3-5 CONCRETE actions the proposal team should take (specific, not vague)
3. List 5-8 exact text strings / phrases that SHOULD appear in a revised proposal
4. Rate fix difficulty: "easy" (add/rephrase), "medium" (restructure), "hard" (genuine gap)

Return ONLY valid JSON:
{{
  "root_cause": "<one sentence>",
  "fix_actions": [
    "<concrete action 1 — be specific about what document/section/table to add>",
    "<concrete action 2>",
    "<concrete action 3>"
  ],
  "evidence_keywords": [
    "<exact phrase that should appear in proposal>",
    "<another phrase>",
    "<another>"
  ],
  "fix_difficulty": "easy|medium|hard"
}}
"""


def _explain_gap_with_llm(score_entry: dict, gap_type: str) -> dict:
    """Generate LLM explanation for a single criterion's gap."""
    prompt = _GAP_EXPLAIN_PROMPT.format(
        parameter      = score_entry.get("parameter", ""),
        max_marks      = score_entry.get("max_marks", 0),
        achieved       = score_entry.get("score", 0),
        marks_lost     = score_entry.get("max_marks", 0) - (score_entry.get("score") or 0),
        gap_type       = gap_type,
        criteria_text  = score_entry.get("criteria_text", "")[:400],
        extracted_value= score_entry.get("extracted_value", "Not found"),
        justification  = score_entry.get("justification", ""),
    )
    raw    = call_llm(prompt, label=f"gap-{score_entry.get('parameter','?')[:20]}")
    result = extract_json(raw) if raw else {}
    return result or {
        "root_cause":        f"Evidence for '{score_entry.get('parameter','')}' was insufficient",
        "fix_actions":       ["Add explicit statement with supporting documents and page references"],
        "evidence_keywords": [],
        "fix_difficulty":    "medium",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_gaps(
    scores:       list[dict],
    doc_max:      int,
    threshold_pct: float = 70.0,
    use_llm:      bool   = True,
) -> GapAnalysis:
    """
    Analyse scored results and produce a structured gap analysis.

    Parameters
    ----------
    scores        : list of score dicts from run_tq_evaluation()
    doc_max       : maximum document-based marks (excl. live assessment)
    threshold_pct : qualifying threshold (default 70%)
    use_llm       : if True, call LLM for per-criterion root cause analysis
    """
    # Filter to document-layer leaf scores only
    leaf = [s for s in scores
            if s.get("evaluation_layer") == "document"
            and s.get("extracted_value") != "Sum of sub-criteria"
            and not s.get("is_parent")
            and s.get("score") is not None]

    total_scored = round(sum(s.get("score") or 0 for s in leaf), 1)
    min_required = round(threshold_pct / 100.0 * doc_max, 1)
    deficit      = round(min_required - total_scored, 1)

    if deficit <= 0:
        risk = "safe" if total_scored >= min_required * 1.15 else "marginal"
    else:
        risk = "at_risk" if deficit <= 10 else "failed"

    gaps: list[CriterionGap] = []

    for s in leaf:
        max_m     = int(s.get("max_marks", 0))
        achieved  = float(s.get("score") or 0)
        marks_lost = round(max_m - achieved, 1)

        if marks_lost <= 0:
            continue   # full marks — no gap

        gap_type   = _classify_gap(s)
        difficulty = _estimate_difficulty(gap_type, marks_lost, s.get("parameter", ""))

        if use_llm and marks_lost >= 3:
            llm_info = _explain_gap_with_llm(s, gap_type)
        else:
            llm_info = {
                "root_cause":        f"Partial evidence for {s.get('parameter','')}",
                "fix_actions":       ["Strengthen evidence with explicit page references"],
                "evidence_keywords": [],
                "fix_difficulty":    difficulty,
            }

        difficulty  = llm_info.get("fix_difficulty", difficulty)
        prio_weight = _DIFFICULTY_WEIGHTS.get(difficulty, 0.5)
        prio_score  = round(marks_lost * prio_weight, 2)

        gaps.append(CriterionGap(
            parameter          = s.get("parameter", ""),
            max_marks          = max_m,
            achieved_marks     = achieved,
            marks_lost         = marks_lost,
            gap_type           = gap_type,
            root_cause         = llm_info.get("root_cause", ""),
            fix_actions        = llm_info.get("fix_actions", []),
            evidence_keywords  = llm_info.get("evidence_keywords", []),
            fix_difficulty     = difficulty,
            priority_score     = prio_score,
            is_sub_item        = bool(s.get("is_sub_item", False)),
            parent_parameter   = s.get("parent_parameter", ""),
        ))

    # Sort by priority (highest marks × easiest first)
    gaps.sort(key=lambda g: g.priority_score, reverse=True)

    recoverable = round(sum(
        g.marks_lost for g in gaps if g.fix_difficulty == "easy"
    ), 1)

    top_n = min(3, len(gaps))
    analysis = GapAnalysis(
        total_scored       = total_scored,
        doc_max            = doc_max,
        threshold_pct      = threshold_pct,
        min_required       = min_required,
        marks_deficit      = deficit,
        qualification_risk = risk,
        gaps               = gaps,
        top_priorities     = gaps[:top_n],
        recoverable_marks  = recoverable,
        summary            = _build_summary(total_scored, doc_max, deficit, risk,
                                            gaps, recoverable),
    )
    return analysis


def _build_summary(total: float, max_m: int, deficit: float, risk: str,
                   gaps: list[CriterionGap], recoverable: float) -> str:
    pct = round(total / max_m * 100, 1) if max_m else 0
    parts = [f"Scored {total}/{max_m} ({pct}%)."]

    if risk == "safe":
        parts.append("Proposal comfortably qualifies.")
    elif risk == "marginal":
        parts.append("Qualifies but with limited margin.")
    elif risk == "at_risk":
        parts.append(f"At risk — {deficit} marks below threshold.")
    else:
        parts.append(f"Does NOT qualify — {deficit} marks below threshold.")

    if recoverable > 0:
        parts.append(f"Up to {recoverable} marks recoverable through "
                     f"documentation improvements (easy fixes).")

    if gaps:
        worst = gaps[0]
        parts.append(f"Biggest gap: {worst.parameter} "
                     f"({worst.marks_lost}/{worst.max_marks} marks lost).")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_gap_report(analysis: GapAnalysis, rfp_title: str = "RFP") -> dict:
    """
    Render a gap analysis as both Markdown and JSON.
    Returns {"markdown": str, "json": str}.
    """
    risk_emoji = {"safe": "✅", "marginal": "⚠️", "at_risk": "🔴", "failed": "❌"}
    diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
    gap_emoji  = {"missing_evidence": "📭", "partial": "📊",
                  "unverified": "❓", "full": "✅"}

    lines = [
        f"# Gap Analysis Report — {rfp_title}",
        "",
        "## Summary",
        "",
        f"{risk_emoji.get(analysis.qualification_risk,'⚠️')} **{analysis.summary}**",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Score (doc) | {analysis.total_scored} / {analysis.doc_max} "
        f"({round(analysis.total_scored/analysis.doc_max*100,1) if analysis.doc_max else 0}%) |",
        f"| Min Required | {analysis.min_required} marks ({analysis.threshold_pct}%) |",
        f"| Deficit / Surplus | "
        f"{'−' if analysis.marks_deficit > 0 else '+'}{abs(analysis.marks_deficit)} marks |",
        f"| Recoverable (easy) | {analysis.recoverable_marks} marks |",
        "",
    ]

    if analysis.top_priorities:
        lines += [
            "## 🎯 Top Priorities (fix these first)",
            "",
        ]
        for i, gap in enumerate(analysis.top_priorities, 1):
            lines += [
                f"### {i}. {gap.parameter}",
                f"**Lost**: {gap.marks_lost}/{gap.max_marks} marks  |  "
                f"**Difficulty**: {diff_emoji.get(gap.fix_difficulty,'?')} {gap.fix_difficulty}  |  "
                f"**Type**: {gap_emoji.get(gap.gap_type,'?')} {gap.gap_type.replace('_',' ')}",
                "",
                f"**Root cause**: {gap.root_cause}",
                "",
                "**Actions:**",
            ]
            for action in gap.fix_actions:
                lines.append(f"- {action}")
            if gap.evidence_keywords:
                lines += [
                    "",
                    "**Keywords that should appear in proposal:**",
                    f"`{'` | `'.join(gap.evidence_keywords[:6])}`",
                ]
            lines.append("")

    if len(analysis.gaps) > len(analysis.top_priorities):
        lines += [
            "## All Gaps",
            "",
            "| Criterion | Lost | Difficulty | Type | Root Cause |",
            "|-----------|------|------------|------|------------|",
        ]
        for gap in analysis.gaps:
            lines.append(
                f"| {gap.parameter[:40]} | {gap.marks_lost}/{gap.max_marks} | "
                f"{gap.fix_difficulty} | {gap.gap_type.replace('_',' ')} | "
                f"{gap.root_cause[:60]}... |"
            )
        lines.append("")

    markdown = "\n".join(lines)

    # JSON output
    json_out = json.dumps({
        "summary":          analysis.summary,
        "total_scored":     analysis.total_scored,
        "doc_max":          analysis.doc_max,
        "min_required":     analysis.min_required,
        "marks_deficit":    analysis.marks_deficit,
        "risk":             analysis.qualification_risk,
        "recoverable":      analysis.recoverable_marks,
        "gaps": [g.to_dict() for g in analysis.gaps],
    }, indent=2)

    return {"markdown": markdown, "json": json_out}


# ─────────────────────────────────────────────────────────────────────────────
# Quick runner (for testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate the actual scores from the log output
    mock_scores = [
        {"parameter": "Financial Turnover",             "max_marks": 15, "score": 15.0,
         "evaluation_layer": "document", "evidence_found": True, "verified": True,
         "extracted_value": "480.23 Cr", "justification": "300 Cr+ → 15 marks", "criteria_text": ""},
        {"parameter": "Area of Experience",             "max_marks": 10, "score": 10.0,
         "evaluation_layer": "document", "evidence_found": True, "verified": True,
         "extracted_value": "more than 6 years", "justification": "6+ → 10 marks", "criteria_text": ""},
        {"parameter": "Handling Large Scale Projects",  "max_marks": 15, "score": 15.0,
         "evaluation_layer": "document", "evidence_found": True, "verified": True,
         "extracted_value": "5 large-scale projects", "justification": ">4 → 15 marks", "criteria_text": ""},
        {"parameter": "Manpower",                       "max_marks": 10, "score": 10.0,
         "evaluation_layer": "document", "evidence_found": True, "verified": True,
         "extracted_value": "more than 1000 personnel", "justification": ">1000 → 10 marks", "criteria_text": ""},
        {"parameter": "TSA Experience",                 "max_marks": 10, "score": 5.0,
         "evaluation_layer": "document", "evidence_found": True, "verified": True,
         "extracted_value": "3 projects", "justification": "1-4 projects → 5 marks",
         "criteria_text": "1 to 4 projects: 5 marks. More than 4 projects: 10 marks.", "is_sub_item": True, "parent_parameter": "DDU-GKY Experience"},
        {"parameter": "TSA/PMU/PMC Experience in Maharashtra", "max_marks": 10, "score": 10.0,
         "evaluation_layer": "document", "evidence_found": True, "verified": True,
         "extracted_value": "One or more project", "justification": "Binary → 10 marks",
         "criteria_text": "One or more project: 10 marks.", "is_sub_item": True, "parent_parameter": "DDU-GKY Experience"},
    ]

    analysis = analyze_gaps(mock_scores, doc_max=70, threshold_pct=70, use_llm=False)
    report   = render_gap_report(analysis, rfp_title="DDU-GKY Pune/Amravati/Nagpur")

    print(report["markdown"])
    print("\n--- JSON ---")
    print(report["json"][:500])
