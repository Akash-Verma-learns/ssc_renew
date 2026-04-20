"""
apply_fix.py
============
Run this once from the project root to patch core/proposal_analyzer.py.

    python apply_fix.py [path/to/core/proposal_analyzer.py]

Safe to re-run — idempotent.
"""

import sys
import re
from pathlib import Path

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("core/proposal_analyzer.py")

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found. Pass the correct path as argument.")
    sys.exit(1)

src = TARGET.read_text(encoding="utf-8")

if "_remap_bulk_extractions" in src:
    print("Patch already applied — nothing to do.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Insert normalisation helpers + _remap_bulk_extractions after imports
# ─────────────────────────────────────────────────────────────────────────────

HELPERS = '''
# ── Parameter-name fuzzy matching (patch) ─────────────────────────────────────

def _normalise_param(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    import re as _re
    return _re.sub(r"\\s+", " ", _re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()

def _word_overlap(a: str, b: str) -> float:
    wa = set(_normalise_param(a).split())
    wb = set(_normalise_param(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def _best_criterion_match(llm_param: str, criterion_params: list) -> "Optional[str]":
    norm = _normalise_param(llm_param)
    for cp in criterion_params:
        if _normalise_param(cp) == norm:
            return cp
    scores = [(cp, _word_overlap(llm_param, cp)) for cp in criterion_params]
    scores.sort(key=lambda x: -x[1])
    if scores and scores[0][1] >= 0.30:
        return scores[0][0]
    return None

def _remap_bulk_extractions(raw_extractions: list, band_criteria: list) -> dict:
    """Remap LLM-returned parameter names to canonical criterion names."""
    criterion_params = [c["parameter"] for c in band_criteria]
    result: dict = {}
    for item in raw_extractions:
        llm_param = (item.get("parameter") or "").strip()
        if not llm_param:
            continue
        canonical = _best_criterion_match(llm_param, criterion_params) or llm_param
        entry = {
            "found": bool(item.get("found")),
            "value": item.get("value"),
            "page":  item.get("page"),
        }
        result[canonical] = entry
        status = "✓" if entry["found"] else "✗"
        note = f" [← \\'{llm_param}\\']" if canonical != llm_param else ""
        print(f"  {status} {canonical[:50]:50s} → {str(entry.get('value',''))[:35]}{note}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
'''

# Insert after the last "import" block (before first function/class def)
insert_after = re.search(r'\n(?=\n_CACHE_DIR)', src)
if insert_after:
    pos = insert_after.start()
    src = src[:pos] + "\n" + HELPERS + src[pos:]
    print("✓ Inserted normalisation helpers")
else:
    # Fallback: insert after the try/except fitz block
    insert_after = re.search(r'(_FITZ_OK = False\n)', src)
    if insert_after:
        pos = insert_after.end()
        src = src[:pos] + "\n" + HELPERS + src[pos:]
        print("✓ Inserted normalisation helpers (fallback position)")
    else:
        print("WARNING: Could not find insertion point for helpers — inserting at top")
        src = HELPERS + src

# ─────────────────────────────────────────────────────────────────────────────
# 2. Replace the result-building loop in bulk_extract_values()
# ─────────────────────────────────────────────────────────────────────────────

OLD_LOOP = '''\
    result: dict[str, dict] = {}
    for item in parsed.get("extractions", []):
        param = item.get("parameter", "")
        if param:
            result[param] = {
                "found": bool(item.get("found")),
                "value": item.get("value"),
                "page":  item.get("page"),
            }
            status = "✓" if item.get("found") else "✗"
            print(f"  {status} {param[:50]:50s} → {str(item.get('value',''))[:35]}")

    return result'''

NEW_LOOP = '''\
    return _remap_bulk_extractions(parsed.get("extractions", []), band_criteria)'''

if OLD_LOOP in src:
    src = src.replace(OLD_LOOP, NEW_LOOP, 1)
    print("✓ Replaced bulk_extract_values result loop")
else:
    print("WARNING: Could not find old result loop — check manually")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Add fuzzy fallback to ProposalAnalysis class methods
# ─────────────────────────────────────────────────────────────────────────────

OLD_GET_VALUE = '''\
    def get_value(self, parameter: str) -> Optional[str]:
        hit = self._values.get(parameter)
        return hit.get("value") if hit else None

    def get_found(self, parameter: str) -> bool:
        hit = self._values.get(parameter)
        return bool(hit.get("found")) if hit else False

    def get_page(self, parameter: str) -> Optional[int]:
        hit = self._values.get(parameter)
        return hit.get("page") if hit else None'''

NEW_GET_VALUE = '''\
    def _resolve_key(self, parameter: str) -> "Optional[str]":
        if parameter in self._values:
            return parameter
        return _best_criterion_match(parameter, list(self._values.keys()))

    def get_value(self, parameter: str) -> Optional[str]:
        key = self._resolve_key(parameter)
        hit = self._values.get(key) if key else None
        return hit.get("value") if hit else None

    def get_found(self, parameter: str) -> bool:
        key = self._resolve_key(parameter)
        hit = self._values.get(key) if key else None
        return bool(hit.get("found")) if hit else False

    def get_page(self, parameter: str) -> Optional[int]:
        key = self._resolve_key(parameter)
        hit = self._values.get(key) if key else None
        return hit.get("page") if hit else None'''

if OLD_GET_VALUE in src:
    src = src.replace(OLD_GET_VALUE, NEW_GET_VALUE, 1)
    print("✓ Added _resolve_key fuzzy fallback to ProposalAnalysis")
else:
    print("WARNING: Could not find ProposalAnalysis.get_value — check manually")

# ─────────────────────────────────────────────────────────────────────────────
# Write patched file
# ─────────────────────────────────────────────────────────────────────────────

backup = TARGET.with_suffix(".py.bak")
TARGET.rename(backup)
TARGET.write_text(src, encoding="utf-8")
print(f"\n✅ Patch applied to {TARGET}")
print(f"   Original backed up to {backup}")
print("\nRestart the server for changes to take effect.")
