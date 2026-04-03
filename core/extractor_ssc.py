"""
core/extractor.py
-----------------
SSC1 clause extractor — now uses core.llm_client (Gemini Flash primary).

CHANGES FROM PREVIOUS VERSION
------------------------------
- Replaced direct Ollama REST calls with call_llm() from llm_client.py
- Gemini Flash is now primary (faster, more reliable JSON output)
- Ollama is automatic fallback (no code change needed at call sites)
- DB session handling is unchanged (still opens fresh session per clause)
- All prompts and extraction logic unchanged
"""

import json
import re
from typing import Optional

from core.vector_store import retrieve
from core.llm_client import call_llm, extract_json


# ──────────────────────────────────────────────────────────────────────────────
# GTBL context (injected into eligibility clause prompt)
# ──────────────────────────────────────────────────────────────────────────────

GTBL_CONTEXT = """
IMPORTANT CONTEXT ABOUT GTBL (the bidding firm):
- GTBL was blacklisted/debarred from October 2021 to September 2024.
- As of today, GTBL faces a penalty for non-performance.
- GTBL was previously terminated by a client for contractual breach/unsatisfactory performance.
  This has since been converted to an amicable closure effective 09.01.2026.
Use this factual position when evaluating eligibility declarations.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPTS = {

    "liability": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim liability clause text",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "cap_info": "description of liability cap — e.g. 'contract value', 'uncapped', '2x contract value'",
  "is_uncapped": true or false,
  "notes": "additional observations"
}}
""",

    "insurance": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim insurance clause",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "client_is_coinsured": true or false,
  "requires_client_approval": true or false,
  "flags": ["high-risk conditions"],
  "notes": "additional observations"
}}
""",

    "scope": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim scope of work",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "summary": "3-5 sentence summary of required work",
  "high_risk_activities": ["civil works, DPR, supervision, third-party verification, legal services, AI decision-making, safety of lives, approving grants/payments"],
  "firm_type_required": "consulting firm / audit firm / architectural firm / other",
  "notes": "additional observations"
}}
""",

    "payment": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim payment terms",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "payment_structure": "milestone-based / deliverable-based / deployment-based / monthly / quarterly / annual / mixed",
  "invoice_to_payment_days": <number or null>,
  "has_invoice_cycle": true or false,
  "deliverable_approval_days": <number or null>,
  "has_approval_timeline": true or false,
  "notes": "additional observations"
}}
""",

    "deliverables": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "deliverables and timeline summary",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "deliverables_list": ["deliverable with timeline"],
  "flags": ["overlapping deliverables, unclear acceptance criteria, aggressive timelines, missing client dependencies"],
  "issues": "overall deliverable risk assessment",
  "notes": "additional observations"
}}
""",

    "personnel": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim personnel/staffing clause",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "replacement_days": <number or null>,
  "replacement_conditions": "conditions for replacement",
  "penalties_for_non_compliance": "any penalties",
  "notes": "additional observations"
}}
""",

    "ld": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim liquidated damages clause",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "ld_cap_text": "description — e.g. '10% of contract value', 'uncapped'",
  "ld_cap_percentage": <number or null>,
  "ld_triggers": ["scenarios where LDs apply"],
  "is_uncapped": true or false,
  "notes": "additional observations"
}}
""",

    "penalties": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim penalty clause",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "ld_cap_text": "penalty cap description",
  "ld_cap_percentage": <number or null>,
  "ld_triggers": ["scenarios where penalties apply"],
  "is_uncapped": true or false,
  "notes": "additional observations"
}}
""",

    "termination": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim termination clauses",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "client_termination_rights": "client termination rights",
  "gtbl_termination_rights": "consultant termination rights (if any)",
  "gtbl_can_terminate": true or false,
  "is_unilateral": true or false,
  "recovery_of_past_payments": true or false,
  "notes": "additional observations"
}}
""",

    "eligibility": """
You are a legal contract analyst.
{gtbl_context}
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text": "verbatim eligibility clause",
  "clause_reference": "clause number/section",
  "page_no": "page number or null",
  "declaration_type": "blacklisting / termination / penalty / combined",
  "uses_historical_language": true or false,
  "historical_language_examples": ["exact phrases like 'has not been blacklisted'"],
  "is_no_deviation": true or false,
  "conflicts_with_gtbl_position": true or false,
  "suggested_deviation": "if historical language, suggest change to 'is not' / 'as on date' language",
  "notes": "additional observations"
}}
""",
}


RAG_QUERIES = {
    "liability":    ["limitation of liability clause", "liability cap contract value",
                     "unlimited liability indemnification"],
    "insurance":    ["insurance requirements clause", "co-insured professional indemnity",
                     "insurance policies approval"],
    "scope":        ["scope of work services", "terms of reference deliverables",
                     "scope of assignment consultant"],
    "payment":      ["payment terms invoice", "payment schedule milestone fees",
                     "invoice payment cycle days"],
    "deliverables": ["deliverables submission timeline", "reports deliverables schedule",
                     "acceptance of deliverables criteria"],
    "personnel":    ["key personnel replacement substitution", "staff replacement period",
                     "personnel change requirements"],
    "ld":           ["liquidated damages clause", "LD delay penalty contract value",
                     "liquidated damages cap percentage"],
    "penalties":    ["penalty clause non-performance", "penalties breach of contract",
                     "financial penalties triggers"],
    "termination":  ["termination clause rights", "termination for convenience default",
                     "contract termination consultant"],
    "eligibility":  ["eligibility criteria blacklisting debarment",
                     "declaration undertaking sanctioned",
                     "no adverse record eligibility",
                     "termination penalty declaration",
                     "no deviation clause unconditional acceptance"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Learning context (fresh DB session per call)
# ──────────────────────────────────────────────────────────────────────────────

def _get_learning_context(clause_type: str, offering: str, solution: str) -> str:
    from database import SessionLocal
    from rules.learning_store import build_fewshot_context
    db = SessionLocal()
    try:
        return build_fewshot_context(
            clause_type=clause_type, offering=offering,
            solution=solution, db=db,
        ) or ""
    except Exception as e:
        print(f"    [Learning] context skipped ({clause_type}): {e}")
        return ""
    finally:
        try:
            db.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_clause(
    clause_type:      str,
    doc_name:         str,
    top_k:            int = 6,
    learning_context: str = "",
) -> dict:
    if clause_type not in EXTRACTION_PROMPTS:
        raise ValueError(f"Unknown clause type '{clause_type}'.")

    # Multi-query RAG retrieval
    queries    = RAG_QUERIES.get(clause_type, [clause_type])
    seen_ids   = set()
    all_chunks = []
    for query in queries:
        for chunk in retrieve(query, doc_name=doc_name, top_k=3):
            cid = chunk["clause_ref"] + str(chunk["page_no"])
            if cid not in seen_ids and chunk["score"] > 0.25:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    all_chunks = all_chunks[:top_k]

    if not all_chunks:
        return {
            "clause_type": clause_type, "doc_name": doc_name,
            "retrieved_chunks": [],
            "extracted": {"clause_text": None, "clause_reference": "Not found", "page_no": None},
            "error": "No relevant chunks found.", "learning_applied": False,
        }

    context = "\n\n---\n\n".join(
        f"[Page {c['page_no']} | {c['section_heading']} | Ref: {c['clause_ref']}]\n{c['text']}"
        for c in all_chunks
    )

    template = EXTRACTION_PROMPTS[clause_type]
    prompt   = (
        template.format(context=context, gtbl_context=GTBL_CONTEXT,
                        learning_context=learning_context)
        if clause_type == "eligibility"
        else template.format(context=context, learning_context=learning_context)
    )

    raw      = call_llm(prompt, label=f"extract-{clause_type}")
    parsed   = extract_json(raw)

    if parsed is None:
        return {
            "clause_type": clause_type, "doc_name": doc_name,
            "retrieved_chunks": all_chunks,
            "extracted": {"clause_text": raw[:500] if raw else None,
                          "clause_reference": "Parse error"},
            "error": f"Could not parse LLM output as JSON",
            "learning_applied": bool(learning_context),
        }

    return {
        "clause_type": clause_type, "doc_name": doc_name,
        "retrieved_chunks": all_chunks,
        "extracted": parsed,
        "error": None,
        "learning_applied": bool(learning_context),
    }


def extract_all_clauses(
    doc_name: str,
    offering: str = "",
    solution: str = "",
    db       = None,       # kept for API compatibility, not used
    model:   str = "",     # kept for API compatibility, not used
) -> dict:
    """
    Run extraction for all 10 clause types.
    Fresh DB session per clause (never holds DB open during LLM call).
    """
    results      = {}
    clause_types = list(EXTRACTION_PROMPTS.keys())
    use_learning = bool(offering or solution)

    print(f"\n[Extractor] Processing {len(clause_types)} clauses for '{doc_name}'...")
    if use_learning:
        print(f"  [Learning] Injecting few-shot context: {offering!r} / {solution!r}")

    for i, ctype in enumerate(clause_types, 1):
        learning_ctx = ""
        if use_learning:
            learning_ctx = _get_learning_context(ctype, offering, solution)

        print(f"  [{i}/{len(clause_types)}] Extracting: {ctype}...")
        results[ctype] = extract_clause(ctype, doc_name,
                                        learning_context=learning_ctx)
        if results[ctype]["error"]:
            print(f"    ⚠ {results[ctype]['error']}")
        else:
            print(f"    ✓ Done")

    return results