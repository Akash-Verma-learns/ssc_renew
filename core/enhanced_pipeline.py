"""
core/enhanced_pipeline.py
==========================
Enhanced Pipeline Orchestrator
================================

Integrates:
  1. Multi-strategy parser  (LlamaParse → Mistral OCR → Docling → PyMuPDF)
  2. Knowledge graph        (corporate term disambiguation + risk ontology)
  3. PageIndex retriever    (vectorless, reasoning-based retrieval)
  4. ChromaDB vector store  (semantic similarity fallback)
  5. LLM extraction         (Gemini primary, Ollama fallback)

Data flow:
  PDF/DOCX
    ↓ document_parser.py       (multi-strategy, best available)
    ↓ knowledge_graph.py       (entity extraction + risk ontology)
    ↓ [ChromaDB ingest]        (vector fallback)
    ↓ pageindex_retriever.py   (tree index build, vectorless retrieval)
    ↓ [For each clause type]:
        ├── PageIndex retrieve (primary: reasoning-based)
        ├── ChromaDB retrieve  (fallback: similarity-based)
        ├── KG context inject  (abbreviations + thresholds)
        └── LLM extraction     (Gemini / Ollama)
    ↓ risk_engine.py           (deterministic rule evaluation)
    ↓ DOCX output

Usage:
    from core.enhanced_pipeline import run_enhanced_pipeline

    result = run_enhanced_pipeline(
        rfp_path="tender.pdf",
        output_path="filled_ssc1.docx",
        offering="AGRI & ALLIED",
        solution="PROGRAMME MANAGEMENT - AGRI & ALLIED",
    )
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.document_parser import parse_document, Chunk
from core.knowledge_graph import build_knowledge_graph, get_prompt_context, KnowledgeGraph
from core.pageindex_retriever import PageIndexRetriever, register_retriever, pageindex_retrieve

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

CLAUSE_ORDER = [
    "liability", "insurance", "scope", "payment", "deliverables",
    "personnel", "ld", "penalties", "termination", "eligibility",
]

CLAUSE_DISPLAY_NAMES = {
    "liability":    "Limitation of Liability",
    "insurance":    "Insurance Clause",
    "scope":        "Scope of Work",
    "payment":      "Payment Terms",
    "deliverables": "Deliverables",
    "personnel":    "Replacement/Substitution of Personnel/Key Resources",
    "ld":           "Liquidated Damages",
    "penalties":    "Penalties",
    "termination":  "Termination Rights",
    "eligibility":  "Eligibility Clause",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main enhanced pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_enhanced_pipeline(
    rfp_path:      str,
    output_path:   str  = "",
    template_path: str  = "document_for_format.docx",
    offering:      str  = "",
    solution:      str  = "",
    progress_cb    = None,
) -> dict:
    """
    Full enhanced pipeline: parse → graph → index → extract → evaluate → write.

    Args:
        rfp_path:      Path to RFP PDF or DOCX
        output_path:   Path for filled SSC1 DOCX (optional)
        template_path: SSC1 template DOCX
        offering:      GT offering (e.g. "AGRI & ALLIED")
        solution:      GT solution
        progress_cb:   Optional callback(step: str, pct: int)

    Returns:
        Result dict with all clause extractions and risk assessments.
    """
    doc_name = Path(rfp_path).name

    def _prog(step: str, pct: int):
        if progress_cb:
            try:
                progress_cb(step, pct)
            except Exception:
                pass
        print(f"[Enhanced] {pct:3d}% — {step}")

    # ── Step 1: Parse document ─────────────────────────────────────────────
    _prog("Parsing document (multi-strategy)", 5)
    chunks, parser_used = parse_document(rfp_path, doc_name=doc_name, verbose=True)
    print(f"[Enhanced] Parser used: {parser_used} | Chunks: {len(chunks)}")

    if not chunks:
        return {"error": "Document parsing produced no chunks", "parser": parser_used}

    # ── Step 2: Build knowledge graph ─────────────────────────────────────
    _prog("Building knowledge graph", 15)
    kg = build_knowledge_graph(chunks, doc_name=doc_name)
    kg_summary = kg.to_summary()
    print(f"[Enhanced] KG: {kg_summary['nodes']} nodes, "
          f"{len(kg_summary['organisations'])} orgs detected")

    # Auto-detect offering if not provided
    if not offering:
        offering = kg.get_offering_hint(chunks)
        if offering:
            print(f"[Enhanced] KG inferred offering: {offering!r}")

    # ── Step 3: Ingest into ChromaDB (vector fallback) ────────────────────
    _prog("Ingesting into vector store (ChromaDB fallback)", 20)
    try:
        from core.vector_store import ingest_chunks as chroma_ingest
        chroma_ingest(chunks, doc_id=doc_name)
        print(f"[Enhanced] ChromaDB: {len(chunks)} chunks ingested")
    except Exception as e:
        print(f"[Enhanced] ChromaDB ingest failed: {e} (PageIndex will cover this)")

    # ── Step 4: Build PageIndex tree ──────────────────────────────────────
    _prog("Building PageIndex tree (vectorless retrieval)", 28)
    retriever = PageIndexRetriever(doc_name=doc_name)
    index_ok  = retriever.build_index(chunks)
    register_retriever(doc_name, retriever)
    if index_ok:
        print(f"[Enhanced] PageIndex tree built successfully")
    else:
        print(f"[Enhanced] PageIndex tree failed — will use ChromaDB only")

    # ── Step 5: Extract clauses ───────────────────────────────────────────
    _prog("Extracting clauses with KG-enriched prompts", 35)
    extraction_results = {}

    for i, clause_type in enumerate(CLAUSE_ORDER):
        pct = 35 + int((i / len(CLAUSE_ORDER)) * 40)
        _prog(f"Extracting: {CLAUSE_DISPLAY_NAMES[clause_type]}", pct)

        result = extract_clause_enhanced(
            clause_type = clause_type,
            doc_name    = doc_name,
            kg          = kg,
            offering    = offering,
            solution    = solution,
        )
        extraction_results[clause_type] = result

        status = "✓" if not result.get("error") else "⚠"
        print(f"  {status} {clause_type}: source={result.get('retrieval_source', '?')}")

    # ── Step 6: Evaluate risk ─────────────────────────────────────────────
    _prog("Evaluating risk", 78)
    from rules.risk_engine import evaluate_clause, RiskResult

    pipeline_results = {}
    for clause_type, ext_result in extraction_results.items():
        exd = ext_result.get("extracted", {})
        try:
            risk = evaluate_clause(clause_type, exd)
        except Exception as e:
            risk = RiskResult(
                clause_name      = clause_type,
                risk_level       = "NEEDS_REVIEW",
                risk_description = f"Evaluation failed: {e}",
                auto_remark      = "",
            )
        pipeline_results[clause_type] = {**ext_result, "risk": risk}
        level = risk.risk_level
        icon  = {"HIGH": "🔴", "MEDIUM": "🟡", "ACCEPTABLE": "🟢",
                  "LOW": "🟢", "NEEDS_REVIEW": "🔵"}.get(level, "⚪")
        print(f"  {icon} {clause_type:15s} → {level}")

    # ── Step 7: Write DOCX (if template exists) ───────────────────────────
    _prog("Writing output", 90)
    if output_path and Path(template_path).exists():
        try:
            from output.writer import build_table_rows, fill_ssc1_table
            table_rows = build_table_rows(pipeline_results)
            fill_ssc1_table(
                rows          = table_rows,
                template_path = template_path,
                output_path   = output_path,
                rfp_name      = doc_name,
            )
            print(f"[Enhanced] Output written: {output_path}")
        except Exception as e:
            print(f"[Enhanced] DOCX write failed: {e}")

    _prog("Done", 100)

    # Summarise
    high_risk   = [k for k, v in pipeline_results.items()
                   if v.get("risk") and v["risk"].risk_level == "HIGH"]
    medium_risk = [k for k, v in pipeline_results.items()
                   if v.get("risk") and v["risk"].risk_level == "MEDIUM"]

    return {
        "doc_name":       doc_name,
        "parser_used":    parser_used,
        "chunk_count":    len(chunks),
        "kg_summary":     kg_summary,
        "offering":       offering,
        "solution":       solution,
        "high_risk":      high_risk,
        "medium_risk":    medium_risk,
        "output_path":    output_path,
        "results":        {
            k: {
                "extracted":        v.get("extracted", {}),
                "risk_level":       v["risk"].risk_level if v.get("risk") else "N/A",
                "risk_description": v["risk"].risk_description if v.get("risk") else "",
                "retrieval_source": v.get("retrieval_source", ""),
            }
            for k, v in pipeline_results.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced clause extraction
# ─────────────────────────────────────────────────────────────────────────────

# Retrieval queries per clause type (used by both PageIndex and ChromaDB)
RAG_QUERIES = {
    "liability":    ["limitation of liability clause", "liability cap contract value",
                     "unlimited liability indemnification", "liability shall not exceed"],
    "insurance":    ["insurance requirements clause", "co-insured professional indemnity",
                     "insurance policies approval", "public liability insurance"],
    "scope":        ["scope of work services", "terms of reference deliverables",
                     "scope of assignment consultant", "tasks and activities"],
    "payment":      ["payment terms invoice", "payment schedule milestone fees",
                     "invoice payment cycle days", "payment within days of invoice"],
    "deliverables": ["deliverables submission timeline", "reports schedule acceptance",
                     "acceptance criteria deliverables", "draft final report"],
    "personnel":    ["key personnel replacement substitution", "staff replacement period",
                     "personnel change requirements", "replacement of expert"],
    "ld":           ["liquidated damages clause", "LD delay penalty contract value",
                     "liquidated damages cap percentage", "delay damages"],
    "penalties":    ["penalty clause non-performance", "penalties breach of contract",
                     "financial penalties triggers", "deduction penalty"],
    "termination":  ["termination clause rights", "termination for convenience default",
                     "contract termination consultant", "right to terminate"],
    "eligibility":  ["eligibility criteria blacklisting debarment", "declaration sanctioned",
                     "no adverse record eligibility", "blacklist debarment declaration",
                     "no deviation clause unconditional acceptance", "has not been penalised"],
}

# Extraction prompt templates (KG context injected into each)
EXTRACTION_PROMPTS = {
    "liability": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":      "verbatim limitation of liability clause text",
  "clause_reference": "clause number/section (e.g. Clause 4.1)",
  "page_no":          "page number or null",
  "cap_info":         "description of liability cap — e.g. 'contract value', 'uncapped', '2x fees'",
  "is_uncapped":      true or false,
  "notes":            "additional observations"
}}""",

    "insurance": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":             "verbatim insurance clause",
  "clause_reference":        "clause number/section",
  "page_no":                 "page number or null",
  "client_is_coinsured":     true or false,
  "requires_client_approval":true or false,
  "flags":                   ["high-risk conditions found"],
  "notes":                   "additional observations"
}}""",

    "scope": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":        "verbatim scope of work text",
  "clause_reference":   "clause number/section",
  "page_no":            "page number or null",
  "summary":            "3-5 sentence summary of what the consultant must do",
  "high_risk_activities":["civil works, DPR, supervision, third-party verification, legal services, AI decisions, safety of lives, approving grants/payments"],
  "firm_type_required": "consulting firm / audit firm / architectural firm / other",
  "notes":              "additional observations"
}}""",

    "payment": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":               "verbatim payment terms",
  "clause_reference":          "clause number/section",
  "page_no":                   "page number or null",
  "payment_structure":         "milestone-based / deliverable-based / monthly / quarterly / mixed",
  "invoice_to_payment_days":   <number or null>,
  "has_invoice_cycle":         true or false,
  "deliverable_approval_days": <number or null>,
  "has_approval_timeline":     true or false,
  "notes":                     "additional observations"
}}""",

    "deliverables": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":      "deliverables and timeline summary",
  "clause_reference": "clause number/section",
  "page_no":          "page number or null",
  "deliverables_list":["each deliverable with its timeline"],
  "flags":            ["overlapping deliverables, unclear acceptance criteria, aggressive timelines, missing client dependencies"],
  "issues":           "overall deliverable risk assessment",
  "notes":            "additional observations"
}}""",

    "personnel": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":                  "verbatim personnel/staffing clause",
  "clause_reference":             "clause number/section",
  "page_no":                      "page number or null",
  "replacement_days":             <number or null>,
  "replacement_conditions":       "conditions under which replacement is allowed",
  "penalties_for_non_compliance": "any penalties mentioned",
  "notes":                        "additional observations"
}}""",

    "ld": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":        "verbatim liquidated damages clause",
  "clause_reference":   "clause number/section",
  "page_no":            "page number or null",
  "ld_cap_text":        "e.g. '10% of contract value', 'uncapped'",
  "ld_cap_percentage":  <number or null>,
  "ld_triggers":        ["scenarios where LDs apply"],
  "is_uncapped":        true or false,
  "notes":              "additional observations"
}}""",

    "penalties": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":       "verbatim penalty clause",
  "clause_reference":  "clause number/section",
  "page_no":           "page number or null",
  "ld_cap_text":       "penalty cap description",
  "ld_cap_percentage": <number or null>,
  "ld_triggers":       ["scenarios where penalties apply"],
  "is_uncapped":       true or false,
  "notes":             "additional observations"
}}""",

    "termination": """{kg_context}

You are a legal contract analyst. Read the following clauses from an RFP.
{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":                "verbatim termination clauses",
  "clause_reference":           "clause number/section",
  "page_no":                    "page number or null",
  "client_termination_rights":  "describe client's termination rights",
  "gtbl_termination_rights":    "describe consultant termination rights (if any)",
  "gtbl_can_terminate":         true or false,
  "is_unilateral":              true or false,
  "recovery_of_past_payments":  true or false,
  "notes":                      "additional observations"
}}""",

    "eligibility": """{kg_context}

You are a legal contract analyst.

GTBL CONTEXT (the bidding firm):
- GTBL (Grant Thornton Bharat LLP) was blacklisted/debarred Oct 2021–Sep 2024.
- As of today, GTBL faces a penalty for non-performance.
- GTBL was terminated by a client; since converted to amicable closure 09.01.2026.
Use this factual position when evaluating eligibility declarations.

{learning_context}
CLAUSES:
{context}

Return ONLY valid JSON:
{{
  "clause_text":                   "verbatim eligibility clause",
  "clause_reference":              "clause number/section",
  "page_no":                       "page number or null",
  "declaration_type":              "blacklisting / termination / penalty / combined",
  "uses_historical_language":      true or false,
  "historical_language_examples":  ["exact phrases like 'has not been blacklisted'"],
  "is_no_deviation":               true or false,
  "conflicts_with_gtbl_position":  true or false,
  "suggested_deviation":           "if historical language, suggest minimum change to 'is not' / 'as on date'",
  "notes":                         "additional observations"
}}""",
}


def extract_clause_enhanced(
    clause_type: str,
    doc_name:    str,
    kg:          Optional[KnowledgeGraph] = None,
    offering:    str = "",
    solution:    str = "",
    top_k:       int = 6,
) -> dict:
    """
    Extract a single clause using PageIndex + KG-enriched prompts.

    Retrieval priority:
      1. PageIndex (vectorless tree-search) — if index available
      2. ChromaDB (semantic similarity) — always available fallback

    Prompt enrichment:
      - Knowledge graph context (abbreviations, risk thresholds)
      - Learning context from past feedback (if available)
    """
    queries = RAG_QUERIES.get(clause_type, [clause_type])

    # ── Retrieval: try PageIndex first ────────────────────────────────────
    all_chunks  = []
    source_used = "none"

    # PageIndex retrieval
    pi_chunks = []
    for query in queries[:2]:  # top 2 queries for PageIndex
        results = pageindex_retrieve(query, doc_name=doc_name, top_k=3)
        pi_chunks.extend(results)

    if pi_chunks:
        # Deduplicate
        seen      = set()
        unique_pi = []
        for c in pi_chunks:
            key = c.get("text", "")[:80]
            if key not in seen:
                seen.add(key)
                unique_pi.append(c)
        all_chunks  = unique_pi[:top_k]
        source_used = "pageindex"

    # ChromaDB fallback (supplement or replace)
    if len(all_chunks) < 3:
        try:
            from core.vector_store import retrieve as chroma_retrieve
            seen_ids = {c.get("text", "")[:80] for c in all_chunks}
            for query in queries:
                chroma_results = chroma_retrieve(query, doc_name=doc_name, top_k=3)
                for r in chroma_results:
                    key = r.get("text", "")[:80]
                    if key not in seen_ids and r.get("score", 0) > 0.2:
                        seen_ids.add(key)
                        all_chunks.append(r)
            source_used = "pageindex+chroma" if source_used == "pageindex" else "chroma"
        except Exception as e:
            print(f"  [Retrieval] ChromaDB failed: {e}")

    if not all_chunks:
        return {
            "clause_type":     clause_type,
            "doc_name":        doc_name,
            "extracted":       {"clause_text": None, "clause_reference": "Not found"},
            "error":           "No chunks retrieved",
            "retrieval_source": source_used,
        }

    # ── Build context string ──────────────────────────────────────────────
    context_parts = []
    for c in all_chunks[:top_k]:
        pg   = c.get("page_no", "?")
        sec  = c.get("section_heading", c.get("section", ""))
        ref  = c.get("clause_ref", c.get("ref", ""))
        text = c.get("text", "")
        context_parts.append(f"[Page {pg} | {sec} | Ref: {ref}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    # ── Build KG context ──────────────────────────────────────────────────
    kg_context       = get_prompt_context(clause_type, kg)
    learning_context = _get_learning_context(clause_type, offering, solution)

    # ── Build prompt ──────────────────────────────────────────────────────
    template = EXTRACTION_PROMPTS.get(clause_type, "")
    if not template:
        return {"clause_type": clause_type, "error": "Unknown clause type",
                "retrieval_source": source_used}

    prompt = template.format(
        context          = context,
        kg_context       = kg_context,
        learning_context = learning_context,
    )

    # ── Call LLM ─────────────────────────────────────────────────────────
    raw = _call_llm(prompt, label=f"extract-{clause_type}")
    if not raw:
        return {
            "clause_type":      clause_type,
            "doc_name":         doc_name,
            "retrieved_chunks": all_chunks,
            "extracted":        {"clause_text": None, "clause_reference": "LLM error"},
            "error":            "LLM returned empty response",
            "retrieval_source": source_used,
        }

    # ── Parse JSON ────────────────────────────────────────────────────────
    extracted = _parse_json(raw)
    if extracted is None:
        return {
            "clause_type":      clause_type,
            "doc_name":         doc_name,
            "retrieved_chunks": all_chunks,
            "extracted":        {"clause_text": raw[:400], "clause_reference": "Parse error"},
            "error":            "Could not parse LLM output as JSON",
            "retrieval_source": source_used,
        }

    return {
        "clause_type":      clause_type,
        "doc_name":         doc_name,
        "retrieved_chunks": all_chunks,
        "extracted":        extracted,
        "error":            None,
        "retrieval_source": source_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_learning_context(clause_type: str, offering: str, solution: str) -> str:
    try:
        from database import SessionLocal
        from rules.learning_store import build_fewshot_context
        db = SessionLocal()
        try:
            return build_fewshot_context(clause_type, offering, solution, db) or ""
        finally:
            db.close()
    except Exception:
        return ""


def _call_llm(prompt: str, label: str = "") -> str:
    """Call best available LLM (Gemini → Ollama)."""
    if GEMINI_API_KEY:
        try:
            return _call_gemini(prompt)
        except Exception as e:
            print(f"  [{label}] Gemini failed: {e} — trying Ollama")
    return _call_ollama(prompt)


def _call_gemini(prompt: str) -> str:
    import time
    from google import genai
    from google.genai import types
    client   = genai.Client(api_key=GEMINI_API_KEY)
    config   = types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048)
    time.sleep(4.2)  # rate limit
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite", contents=prompt, config=config)
    return response.text or ""


def _call_ollama(prompt: str) -> str:
    import requests
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": OLLAMA_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "options": {"temperature": 0.0, "num_ctx": 4096}},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["message"]["content"] or ""
    except Exception as e:
        print(f"  [Ollama] error: {e}")
        return ""


def _parse_json(text: str) -> Optional[dict]:
    """Robustly parse JSON from LLM output."""
    import json
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = in_str = esc = False
    depth_n = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth_n += 1
        elif ch == "}":
            depth_n -= 1
            if depth_n == 0:
                end = i + 1; break
    candidate = text[start:end] if end > start else text[start:]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        return None
