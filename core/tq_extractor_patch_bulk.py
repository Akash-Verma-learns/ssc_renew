"""
core/tq_extractor_patch_bulk.py
================================
Patch to wire warm_analysis_cache() into run_tq_evaluation()
before the per-criterion scoring loop begins.

APPLY: In run_tq_evaluation(), after ingest_proposal() and before
the for-loop over scoreable criteria, add:

    # Warm proposal analysis cache (1 bulk LLM call total)
    try:
        from core.tq_scorer import warm_analysis_cache
        analysis = warm_analysis_cache(proposal_path, scoreable)
        if analysis:
            n_vals = len([p for p in [analysis.get_value(c['parameter'])
                          for c in scoreable] if p])
            n_cvs  = sum(1 for c in scoreable
                         if (c.get('formula_type','').upper() in ('QUAL','LLM')
                             and analysis.is_cv_present(c['parameter'])))
            _prog(f"Proposal analyzed: {n_vals} values, {n_cvs} CVs detected", 25)
    except Exception as e:
        print(f"[TQ] Warm analysis cache error (non-fatal): {e}")

HOW TO APPLY THIS PATCH
────────────────────────
In tq_extractor.py, find this line in run_tq_evaluation():

    ingest_proposal(proposal_path, proposal_doc_name)
    _prog("Scoring criteria against proposal", 28)

Replace with:

    ingest_proposal(proposal_path, proposal_doc_name)

    # ── Warm proposal analysis cache ──────────────────────────────────────
    _prog("Analyzing proposal (bulk extraction)", 23)
    try:
        from core.tq_scorer import warm_analysis_cache
        warm_analysis_cache(proposal_path, scoreable)
        _prog("Scoring criteria against proposal", 28)
    except Exception as e:
        print(f"[TQ] Analysis warm error (non-fatal): {e}")
        _prog("Scoring criteria against proposal", 28)
    # ──────────────────────────────────────────────────────────────────────

This ensures ONE bulk LLM call extracts all values before the per-criterion
scoring loop, reducing total LLM calls from 28 to 1-3.
"""

# This file is documentation only — apply the patch manually to tq_extractor.py
# OR use the apply_patch() function below from your shell:

def apply_patch(tq_extractor_path: str = "core/tq_extractor.py") -> bool:
    """
    Automatically apply the warm_analysis_cache patch to tq_extractor.py.
    Safe to run multiple times (idempotent).
    """
    from pathlib import Path

    path = Path(tq_extractor_path)
    if not path.exists():
        print(f"File not found: {tq_extractor_path}")
        return False

    content = path.read_text(encoding="utf-8")

    # Don't apply twice
    if "warm_analysis_cache" in content:
        print("Patch already applied.")
        return True

    old = '''    ingest_proposal(proposal_path, proposal_doc_name)
    _prog("Scoring criteria against proposal", 28)'''

    new = '''    ingest_proposal(proposal_path, proposal_doc_name)

    # ── Warm proposal analysis cache (1 bulk LLM call total) ──────────────
    _prog("Analyzing proposal (bulk extraction)", 23)
    try:
        from core.tq_scorer import warm_analysis_cache
        warm_analysis_cache(proposal_path, scoreable)
    except Exception as e:
        print(f"[TQ] Analysis warm error (non-fatal): {e}")
    _prog("Scoring criteria against proposal", 28)
    # ──────────────────────────────────────────────────────────────────────'''

    if old not in content:
        print("Could not find patch target — apply manually (see file docstring)")
        return False

    patched = content.replace(old, new, 1)
    path.write_text(patched, encoding="utf-8")
    print(f"✓ Patch applied to {tq_extractor_path}")
    return True


if __name__ == "__main__":
    apply_patch()
