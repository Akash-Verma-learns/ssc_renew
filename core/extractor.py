"""
Clause Extractor
----------------
Uses Ollama (local, free) to extract structured data from RFP clause text.

WINDOWS FIX
-----------
The ollama Python client (0.2.1) uses httpx internally. On Windows, httpx
can stall indefinitely waiting for a socket close signal even after the full
response body has been received. This manifests as the pipeline hanging after
"[N/10] Extracting: <clause>..." with no further output.

Fix: call Ollama's REST API directly via `requests` (standard library HTTP
client that handles Windows sockets reliably) with an explicit timeout.
The ollama import is kept only for the ResponseError type; the actual HTTP
call no longer goes through ollama.Client.chat().

DB SESSION FIX (v2)
-------------------
Root cause of the SSL-connection-closed cascade:
  1. Pipeline opens a DB session and holds it open.
  2. A long Ollama call (sometimes 300–600 s) occupies the thread.
  3. Neon's serverless proxy silently drops idle TCP connections after ~5 min.
  4. The DB session now holds a dead socket.
  5. The next query (build_fewshot_context) raises OperationalError.
  6. SQLAlchemy puts the session into a "transaction failed" state.
  7. Rollback also fails (dead socket) → session is permanently broken.
  8. Every subsequent query in the same session raises
     "Can't reconnect until invalid transaction is rolled back".
  9. db.close() at the end raises the same error → uvicorn logs ASGI error.

Fix: extract_all_clauses now opens a *fresh*, short-lived DB session for
each learning-context lookup. The short session is closed immediately after
the query, so it is never held open during an Ollama call. The caller's
session (used for RFP state updates) is never touched here.
"""

import json
import re
import requests
from typing import Optional
from core.vector_store import retrieve


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_MODEL    = "llama3.2"
OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
# 10 minutes — generous ceiling for a slow CPU / large context window.
# The previous 300 s limit caused ReadTimeout on the liability clause of several
# RFPs. Even if this is hit, the error is caught and the clause is skipped
# gracefully; the pipeline does NOT crash.
OLLAMA_TIMEOUT  = 600

GTBL_CONTEXT = """
IMPORTANT CONTEXT ABOUT GTBL (the bidding firm):
- GTBL was blacklisted/debarred from October 2021 to September 2024.
- As of today, GTBL faces a penalty for non-performance.
- GTBL was previously terminated by a client for contractual breach/unsatisfactory performance.
  This has since been converted to an amicable closure effective 09.01.2026.
Use this factual position when evaluating eligibility declarations.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates per clause type
# ──────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPTS = {

    "liability": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of the limitation of liability clause",
  "clause_reference": "clause number or section reference (e.g. Clause 4.1)",
  "page_no": "page number if visible, else null",
  "cap_info": "description of the liability cap - e.g. 'contract value', 'uncapped', '2x contract value', '50% of fees'",
  "is_uncapped": true or false,
  "notes": "any additional relevant observations"
}}
If the clause is not found, return {{"clause_text": null, "cap_info": "not found"}}.
""",

    "insurance": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of the insurance clause",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "client_is_coinsured": true or false,
  "requires_client_approval": true or false,
  "flags": ["list any high-risk conditions found"],
  "notes": "any additional relevant observations"
}}
If not found, return {{"clause_text": null}}.
""",

    "scope": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of the scope of work",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "summary": "3-5 sentence summary of what the consultant/firm is required to do",
  "high_risk_activities": [
    "list only activities that are high-risk: civil works, DPR, supervision, third-party verification, legal services, AI decision-making, gambling, safety of lives, approving grants/payments"
  ],
  "firm_type_required": "consulting firm / audit firm / architectural firm / other",
  "notes": "any additional observations"
}}
""",

    "payment": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of the payment terms",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "payment_structure": "milestone-based / deliverable-based / deployment-based / monthly / quarterly / annual / mixed",
  "invoice_to_payment_days": number or null,
  "has_invoice_cycle": true or false,
  "deliverable_approval_days": number or null,
  "has_approval_timeline": true or false,
  "notes": "any additional observations"
}}
""",

    "deliverables": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "summary of deliverables and timelines",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "deliverables_list": ["list each deliverable with its timeline"],
  "flags": [
    "list any of these if found: overlapping deliverables, unclear acceptance criteria, aggressive timelines, missing client dependencies"
  ],
  "issues": "overall assessment of deliverable risks",
  "notes": "any additional observations"
}}
""",

    "personnel": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of the personnel/staffing clause",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "replacement_days": number or null,
  "replacement_conditions": "conditions under which replacement is allowed",
  "penalties_for_non_compliance": "any penalties mentioned",
  "notes": "any additional observations"
}}
""",

    "ld": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of the liquidated damages clause",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "ld_cap_text": "description of LD cap e.g. '10% of contract value', 'uncapped', '20% of fees'",
  "ld_cap_percentage": number or null,
  "ld_triggers": ["scenarios where LDs apply"],
  "is_uncapped": true or false,
  "notes": "any additional observations"
}}
""",

    "penalties": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of the penalty clause",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "ld_cap_text": "description of penalty cap e.g. '10% of contract value', 'uncapped'",
  "ld_cap_percentage": number or null,
  "ld_triggers": ["scenarios where penalties apply"],
  "is_uncapped": true or false,
  "notes": "any additional observations"
}}
""",

    "termination": """
You are a legal contract analyst. Read the following clause(s) from an RFP/tender document.
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of termination clauses",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "client_termination_rights": "describe client's termination rights",
  "gtbl_termination_rights": "describe consultant/firm's termination rights (if any)",
  "gtbl_can_terminate": true or false,
  "is_unilateral": true or false,
  "recovery_of_past_payments": true or false,
  "notes": "any additional observations"
}}
""",

    "eligibility": """
You are a legal contract analyst.
{gtbl_context}
{learning_context}
CLAUSES:
{context}

Extract the following and return ONLY valid JSON (no explanation):
{{
  "clause_text": "verbatim text of the eligibility clause / declaration",
  "clause_reference": "clause number or section reference",
  "page_no": "page number if visible, else null",
  "declaration_type": "blacklisting / termination / penalty / combined",
  "uses_historical_language": true or false,
  "historical_language_examples": ["exact phrases like 'has not been blacklisted'"],
  "is_no_deviation": true or false,
  "conflicts_with_gtbl_position": true or false,
  "suggested_deviation": "if historical language used, suggest minimum change to 'is not' / 'as on date' language",
  "notes": "any additional observations"
}}
""",
}

RAG_QUERIES = {
    "liability": [
        "limitation of liability clause",
        "liability cap contract value",
        "unlimited liability indemnification",
    ],
    "insurance": [
        "insurance requirements clause",
        "co-insured professional indemnity",
        "insurance policies approval",
    ],
    "scope": [
        "scope of work services",
        "terms of reference deliverables",
        "scope of assignment consultant",
    ],
    "payment": [
        "payment terms invoice",
        "payment schedule milestone fees",
        "invoice payment cycle days",
    ],
    "deliverables": [
        "deliverables submission timeline",
        "reports deliverables schedule",
        "acceptance of deliverables criteria",
    ],
    "personnel": [
        "key personnel replacement substitution",
        "staff replacement period",
        "personnel change requirements",
    ],
    "ld": [
        "liquidated damages clause",
        "LD delay penalty contract value",
        "liquidated damages cap percentage",
    ],
    "penalties": [
        "penalty clause non-performance",
        "penalties breach of contract",
        "financial penalties triggers",
    ],
    "termination": [
        "termination clause rights",
        "termination for convenience default",
        "contract termination consultant",
    ],
    "eligibility": [
        "eligibility criteria blacklisting debarment",
        "declaration undertaking sanctioned",
        "no adverse record eligibility",
        "termination penalty declaration",
        "no deviation clause unconditional acceptance",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = text[start:i + 1]
                raw = re.sub(r",\s*([}\]])", r"\1", raw)
                return raw
    return text[start:]


def _call_ollama(model: str, prompt: str) -> str:
    """
    Direct REST call to Ollama — bypasses the ollama Python client entirely.

    Uses `requests` which handles Windows socket behaviour correctly and
    respects the explicit timeout. Returns the raw text content string,
    or raises requests.RequestException / ValueError on failure.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0,"num_ctx": 2048,},
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def _get_learning_context(
    clause_type: str,
    offering: str,
    solution: str,
) -> str:
    """
    Open a *fresh* short-lived DB session, query learning examples, close it.

    This is the key fix: we never hold a DB connection open across an Ollama
    call. Each call to this helper creates and destroys its own connection,
    so a stale Neon socket can never poison the main pipeline session.

    Returns empty string on any failure (learning context is optional).
    """
    from database import SessionLocal
    from rules.learning_store import build_fewshot_context

    db = SessionLocal()
    try:
        ctx = build_fewshot_context(
            clause_type=clause_type,
            offering=offering,
            solution=solution,
            db=db,
        )
        return ctx or ""
    except Exception as e:
        print(f"    [Learning] context skipped ({clause_type}): {e}")
        return ""
    finally:
        # Always close, even if an exception occurred inside the query.
        # pool_pre_ping will give the next caller a fresh connection.
        try:
            db.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Core extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_clause(
    clause_type: str,
    doc_name: str,
    top_k: int = 6,
    model: str = OLLAMA_MODEL,
    learning_context: str = "",
) -> dict:
    if clause_type not in EXTRACTION_PROMPTS:
        raise ValueError(f"Unknown clause type '{clause_type}'.")

    # ── Step 1: Multi-query RAG retrieval ──────────────────────────────────────
    queries    = RAG_QUERIES.get(clause_type, [clause_type])
    seen_ids   = set()
    all_chunks = []

    for query in queries:
        chunks = retrieve(query, doc_name=doc_name, top_k=3)
        for chunk in chunks:
            cid = chunk["clause_ref"] + str(chunk["page_no"])
            if cid not in seen_ids and chunk["score"] > 0.25:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    all_chunks = all_chunks[:top_k]

    if not all_chunks:
        return {
            "clause_type": clause_type,
            "doc_name": doc_name,
            "retrieved_chunks": [],
            "extracted": {
                "clause_text": None,
                "clause_reference": "Not found",
                "page_no": None,
            },
            "error": "No relevant chunks found in document.",
            "learning_applied": False,
        }

    # ── Step 2: Build context string ──────────────────────────────────────────
    context_parts = []
    for c in all_chunks:
        context_parts.append(
            f"[Page {c['page_no']} | {c['section_heading']} | Ref: {c['clause_ref']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ── Step 3: Build prompt ───────────────────────────────────────────────────
    prompt_template = EXTRACTION_PROMPTS[clause_type]
    if clause_type == "eligibility":
        prompt = prompt_template.format(
            context=context,
            gtbl_context=GTBL_CONTEXT,
            learning_context=learning_context,
        )
    else:
        prompt = prompt_template.format(
            context=context,
            learning_context=learning_context,
        )

    # ── Step 4: Call Ollama via direct REST (Windows-safe) ────────────────────
    try:
        raw_output = _call_ollama(model, prompt)
    except Exception as e:
        return {
            "clause_type": clause_type,
            "doc_name": doc_name,
            "retrieved_chunks": all_chunks,
            "extracted": {
                "clause_text": None,
                "clause_reference": "Ollama error",
            },
            "error": f"{type(e).__name__}: {e}",
            "learning_applied": bool(learning_context),
        }

    if not raw_output or not raw_output.strip():
        return {
            "clause_type": clause_type,
            "doc_name": doc_name,
            "retrieved_chunks": all_chunks,
            "extracted": {
                "clause_text": None,
                "clause_reference": "Empty LLM response",
            },
            "error": "Ollama returned an empty response.",
            "learning_applied": bool(learning_context),
        }

    json_str = _clean_json(raw_output)

    if not json_str:
        return {
            "clause_type": clause_type,
            "doc_name": doc_name,
            "retrieved_chunks": all_chunks,
            "extracted": {
                "clause_text": raw_output[:500],
                "clause_reference": "No JSON in response",
            },
            "error": f"No JSON object found in LLM response: {raw_output[:200]}",
            "learning_applied": bool(learning_context),
        }

    try:
        extracted = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {
            "clause_type": clause_type,
            "doc_name": doc_name,
            "retrieved_chunks": all_chunks,
            "extracted": {
                "clause_text": raw_output[:500],
                "clause_reference": "JSON parse error",
            },
            "error": f"Could not parse LLM output as JSON: {e}",
            "learning_applied": bool(learning_context),
        }

    return {
        "clause_type": clause_type,
        "doc_name": doc_name,
        "retrieved_chunks": all_chunks,
        "extracted": extracted,
        "error": None,
        "learning_applied": bool(learning_context),
    }


def extract_all_clauses(
    doc_name: str,
    model: str = OLLAMA_MODEL,
    offering: str = "",
    solution: str = "",
    db=None,          # kept for API compatibility but NO LONGER USED for queries
) -> dict:
    """
    Run extraction for all 10 clause types.

    Learning context is fetched via _get_learning_context(), which opens and
    closes its own fresh DB session for each clause. This means no DB
    connection is ever held open while Ollama is thinking, eliminating the
    SSL-drop cascade entirely.

    The `db` parameter is accepted for backward compatibility but is ignored.
    Callers do not need to change their call signature.
    """
    if not model:
        model = OLLAMA_MODEL

    results      = {}
    clause_types = list(EXTRACTION_PROMPTS.keys())
    use_learning = bool(offering or solution)

    print(f"\n[Extractor] Processing {len(clause_types)} clauses for '{doc_name}'...")
    if use_learning:
        print(f"  [Learning] Will inject few-shot context for: {offering!r} / {solution!r}")

    for i, ctype in enumerate(clause_types, 1):
        # ── Fetch learning context with its own isolated session ──────────────
        learning_ctx = ""
        if use_learning:
            learning_ctx = _get_learning_context(ctype, offering, solution)
            if learning_ctx:
                print(f"    [{i}/{len(clause_types)}] {ctype}: few-shot context injected ✓")

        # ── Run LLM extraction (no DB connection held during this call) ────────
        print(f"  [{i}/{len(clause_types)}] Extracting: {ctype}...")
        results[ctype] = extract_clause(
            ctype, doc_name, model=model, learning_context=learning_ctx
        )

        if results[ctype]["error"]:
            print(f"    ⚠ Warning: {results[ctype]['error']}")
        else:
            print(f"    ✓ Done")

    return results