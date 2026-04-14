"""
core/cache_admin.py  —  Cache management CLI
============================================

Usage (from project root, Windows):
  python -m core.cache_admin list
  python -m core.cache_admin show   <rfp_filename>
  python -m core.cache_admin delete <rfp_filename>
  python -m core.cache_admin delete-all
  python -m core.cache_admin refresh <rfp_filename>

Examples:
  python -m core.cache_admin list
  python -m core.cache_admin show 0c5800be.pdf
  python -m core.cache_admin delete 0c5800be.pdf
  python -m core.cache_admin refresh 0c5800be.pdf
"""

import json
import sys
from pathlib import Path

# --- Minimal bootstrap so this runs standalone ---
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rfp_cache import _CACHE_DIR, _hash_pdf, _cache_path, invalidate_cache

_RFP_SEARCH_DIRS = [
    Path("./uploads"),
    Path("./tq_uploads"),
    Path("./rfp_uploads"),
    Path("."),
]


def _find_pdf(name: str) -> Path | None:
    for d in _RFP_SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def cmd_list():
    files = sorted(_CACHE_DIR.glob("*.json"))
    if not files:
        print("No cached RFP extractions found.")
        return
    print(f"{'Hash':<20} {'RFP File':<35} {'Criteria':>8} {'Grand':>6} {'Cached At'}")
    print("-" * 90)
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            print(f"{d.get('rfp_hash','?'):<20} "
                  f"{d.get('rfp_filename','?'):<35} "
                  f"{len(d.get('criteria',[])): >8} "
                  f"{d.get('grand_total',0): >6} "
                  f"{d.get('cached_at','?')}")
        except Exception:
            print(f"{f.stem:<20} (unreadable)")


def cmd_show(rfp_name: str):
    pdf = _find_pdf(rfp_name)
    if not pdf:
        print(f"PDF not found: {rfp_name}")
        return
    h  = _hash_pdf(str(pdf))
    cp = _cache_path(h)
    if not cp.exists():
        print(f"No cache for {rfp_name} (hash={h})")
        return
    d = json.loads(cp.read_text(encoding="utf-8"))
    print(f"\n{'='*60}")
    print(f"RFP:          {d.get('rfp_filename')}")
    print(f"Hash:         {d.get('rfp_hash')}")
    print(f"Cached at:    {d.get('cached_at')}")
    print(f"Grand total:  {d.get('grand_total')} marks")
    print(f"Live marks:   {d.get('live_marks')} ({d.get('live_label','')})")
    print(f"Doc total:    {d.get('doc_total')}")
    print(f"Threshold:    {d.get('threshold')}%")
    print(f"\nCriteria ({len(d.get('criteria',[]))}):")
    for c in d.get("criteria", []):
        pfx = "    " if c.get("is_sub_item") else "  "
        tag = "[parent]" if c.get("is_parent") else f"[{c.get('formula_type','?')}]"
        print(f"{pfx}[{c.get('item_code','?'):4s}] {c['parameter'][:55]:55s} "
              f"{c['max_marks']:3d} {tag}")
    print(f"\nBands cached for: {list(d.get('bands',{}).keys())}")
    print(f"{'='*60}\n")


def cmd_delete(rfp_name: str):
    pdf = _find_pdf(rfp_name)
    if not pdf:
        print(f"PDF not found: {rfp_name}")
        return
    deleted = invalidate_cache(str(pdf))
    if deleted:
        print(f"Deleted cache for {rfp_name}")
    else:
        print(f"No cache found for {rfp_name}")


def cmd_delete_all():
    files = list(_CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    print(f"Deleted {len(files)} cache files.")


def cmd_refresh(rfp_name: str):
    """Delete cache then trigger re-extraction."""
    cmd_delete(rfp_name)
    print(f"Cache deleted. Next run will re-extract criteria for {rfp_name}.")
    print(f"Or run: python -c \"from core.tq_extractor import extract_marking_table; "
          f"extract_marking_table('{rfp_name}', force_refresh=True)\"")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()
    if cmd == "list":
        cmd_list()
    elif cmd == "show" and len(args) >= 2:
        cmd_show(args[1])
    elif cmd == "delete" and len(args) >= 2:
        cmd_delete(args[1])
    elif cmd == "delete-all":
        inp = input("Delete ALL cached extractions? [y/N] ")
        if inp.lower() == "y":
            cmd_delete_all()
    elif cmd == "refresh" and len(args) >= 2:
        cmd_refresh(args[1])
    else:
        print(__doc__)
