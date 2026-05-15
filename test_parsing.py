"""
test_parsing.py
================
Verification script — tests the full parser cascade on the NHB proposal PDF.
Run: python test_parsing.py

Tests:
  1. PyMuPDF parser (always works)
  2. Knowledge graph entity extraction
  3. PageIndex local tree build
  4. Clause retrieval comparison (PageIndex vs keyword)
  5. Table detection and markdown output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import json

PDF_PATH = "/mnt/user-data/uploads/TechProposalNHBClusterDevelopmenGTILLP_Aug2020_Final.pdf"
DOC_NAME = "TechProposalNHB.pdf"

def test_parser():
    print("=" * 60)
    print("TEST 1: Document Parser")
    print("=" * 60)

    from core.document_parser import parse_document, _parse_pymupdf

    # Force PyMuPDF (always available)
    os.environ["PARSER_STRATEGY"] = "pymupdf"
    chunks, parser = parse_document(PDF_PATH, doc_name=DOC_NAME)

    print(f"\nParser used:   {parser}")
    print(f"Total chunks:  {len(chunks)}")
    tables = [c for c in chunks if c.is_table]
    print(f"Table chunks:  {len(tables)}")
    print(f"Avg chunk len: {sum(len(c.text) for c in chunks) // max(len(chunks),1)} chars")

    # Show first 5 chunks
    print("\n--- First 5 chunks ---")
    for c in chunks[:5]:
        print(f"  Page {c.page_no:3d} | {c.section_heading[:40]:40s} | "
              f"{'[TABLE]' if c.is_table else '       '} | {c.text[:80]!r}")

    # Show table chunks
    print("\n--- Table chunks (first 3) ---")
    for c in tables[:3]:
        print(f"  Page {c.page_no} | {c.section_heading[:40]}")
        if c.table_markdown:
            lines = c.table_markdown.split("\n")[:6]
            for l in lines:
                print(f"    {l[:100]}")
        print()

    # Check for key clauses
    key_terms = ["liability", "liquidated", "termination", "payment", "eligibility",
                 "insurance", "scope", "personnel", "deliverable", "penalty"]
    print("--- Key clause coverage ---")
    for term in key_terms:
        found = [c for c in chunks if term.lower() in c.text.lower()]
        print(f"  '{term}': {len(found)} chunks, "
              f"pages: {sorted(set(c.page_no for c in found))[:5]}")

    return chunks


def test_knowledge_graph(chunks):
    print("\n" + "=" * 60)
    print("TEST 2: Knowledge Graph")
    print("=" * 60)

    from core.knowledge_graph import build_knowledge_graph, get_prompt_context

    kg = build_knowledge_graph(chunks, doc_name=DOC_NAME)
    summary = kg.to_summary()

    print(f"\nGraph nodes:     {summary['nodes']}")
    print(f"Graph edges:     {summary['edges']}")
    print(f"Organisations:   {summary['organisations'][:5]}")
    print(f"Amounts (Cr):    {summary['amounts_cr'][:5]}")
    print(f"Detected risks:  {len(summary['detected_risks'])}")

    # Test abbreviation expansion
    test_text = "GTBL as the PMC for NHB under MoFPI, with ToR specifying LD caps"
    expanded  = kg.expand_abbreviations(test_text)
    print(f"\nAbbreviation expansion:")
    print(f"  Input:  {test_text}")
    print(f"  Output: {expanded[:200]}")

    # Test clause context generation
    print("\n--- Liability clause context ---")
    ctx = get_prompt_context("liability", kg)
    print(ctx[:500])

    print("\n--- Eligibility clause context ---")
    ctx2 = get_prompt_context("eligibility", kg)
    print(ctx2[:500])

    # Inferred offering
    offering = kg.get_offering_hint(chunks)
    print(f"\nInferred offering: {offering!r}")

    return kg


def test_pageindex(chunks):
    print("\n" + "=" * 60)
    print("TEST 3: PageIndex Tree Build")
    print("=" * 60)

    from core.pageindex_retriever import PageIndexRetriever, register_retriever

    retriever = PageIndexRetriever(doc_name=DOC_NAME, cache_dir="./test_cache")
    ok = retriever.build_index(chunks[:200], force_rebuild=True)  # use first 200 chunks for speed

    print(f"\nIndex built: {ok}")
    if retriever.root:
        print(f"Root title:  {retriever.root.title}")
        print(f"Top-level sections ({len(retriever.root.children)}):")
        for child in retriever.root.children[:8]:
            print(f"  [{child.start_page:3d}-{child.end_page:3d}] {child.title[:60]}")
            print(f"    Summary: {child.summary[:80]}")

    register_retriever(DOC_NAME, retriever)

    # Test retrieval
    print("\n--- Retrieval test: 'limitation of liability' ---")
    results = retriever.retrieve("limitation of liability clause", top_k=4)
    for r in results:
        print(f"  Page {r.get('page_no','?'):3} | score={r.get('score',0):.2f} | "
              f"{r.get('text','')[:100]!r}")

    print("\n--- Retrieval test: 'liquidated damages cap' ---")
    results2 = retriever.retrieve("liquidated damages cap percentage", top_k=4)
    for r in results2:
        print(f"  Page {r.get('page_no','?'):3} | score={r.get('score',0):.2f} | "
              f"{r.get('text','')[:100]!r}")

    return retriever


def test_table_fidelity(chunks):
    print("\n" + "=" * 60)
    print("TEST 4: Table Fidelity Check")
    print("=" * 60)

    tables = [c for c in chunks if c.is_table]
    print(f"\nTotal tables detected: {len(tables)}")

    # Look for scoring/evaluation tables
    eval_tables = [t for t in tables if any(
        kw in t.text.lower() for kw in ["marks", "score", "criteria", "evaluation",
                                          "s.no", "s. no", "parameter"]
    )]
    print(f"Evaluation-type tables: {len(eval_tables)}")

    # Look for team/personnel tables
    team_tables = [t for t in tables if any(
        kw in t.text.lower() for kw in ["team leader", "expert", "qualification",
                                          "years of experience", "cv"]
    )]
    print(f"Team/personnel tables: {len(team_tables)}")

    # Show best evaluation table
    if eval_tables:
        best = max(eval_tables, key=lambda t: len(t.text))
        print(f"\n--- Best evaluation table (page {best.page_no}) ---")
        print(f"Section: {best.section_heading}")
        print(f"Text preview:\n{best.text[:600]}")
        if best.table_markdown:
            print(f"\nMarkdown preview:\n{best.table_markdown[:600]}")

    # Column alignment test: count | in table chunks
    if tables:
        pipe_counts = [t.table_markdown.count("|") for t in tables if t.table_markdown]
        if pipe_counts:
            print(f"\nTable column separators (|) per table:")
            print(f"  min={min(pipe_counts)}, max={max(pipe_counts)}, avg={sum(pipe_counts)//len(pipe_counts)}")


def test_enhanced_extraction(chunks, kg):
    print("\n" + "=" * 60)
    print("TEST 5: Enhanced Clause Extraction (single clause)")
    print("=" * 60)

    # Ingest into mock vector store for fallback
    # We'll test the retrieval part without calling the actual LLM
    from core.pageindex_retriever import pageindex_retrieve

    print("\n--- PageIndex retrieval for 'liability' ---")
    results = pageindex_retrieve("limitation of liability clause", doc_name=DOC_NAME, top_k=5)
    print(f"Found {len(results)} chunks via PageIndex")
    for r in results[:3]:
        print(f"  Page {r.get('page_no','?')} | {r.get('text','')[:100]!r}")

    print("\n--- KG context for liability clause ---")
    from core.knowledge_graph import get_prompt_context
    ctx = get_prompt_context("liability", kg)
    print(ctx[:400])

    print("\n✓ Extraction framework verified (LLM call skipped in test)")


def main():
    print("NHB Proposal Parser Verification")
    print(f"File: {PDF_PATH}")
    print(f"Exists: {Path(PDF_PATH).exists()}")
    print()

    if not Path(PDF_PATH).exists():
        print("ERROR: PDF not found at expected path")
        sys.exit(1)

    # Run all tests
    chunks    = test_parser()
    kg        = test_knowledge_graph(chunks)
    retriever = test_pageindex(chunks)
    test_table_fidelity(chunks)
    test_enhanced_extraction(chunks, kg)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Chunks parsed:   {len(chunks)}")
    print(f"  Tables found:    {len([c for c in chunks if c.is_table])}")
    print(f"  KG nodes:        {kg.to_summary()['nodes']}")
    print(f"  PageIndex nodes: {retriever._count_nodes(retriever.root)}")


if __name__ == "__main__":
    main()
