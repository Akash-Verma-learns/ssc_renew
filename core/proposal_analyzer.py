"""
core/proposal_analyzer.py  —  v1  Single-Pass Bulk Extraction
==============================================================
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import fitz
    _FITZ_OK = True
except ImportError:
    _FITZ_OK = False

# ── Parameter-name fuzzy matching (patch) ─────────────────────────────────────

def _normalise_param(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    import re as _re
    return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()

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
        note = f" [← \'{llm_param}\']" if canonical != llm_param else ""
        print(f"  {status} {canonical[:50]:50s} → {str(entry.get('value',''))[:35]}{note}")
    return result

# ─────────────────────────────────────────────────────────────────────────────


_CACHE_DIR = Path("./proposal_cache")
_CACHE_DIR.mkdir(exist_ok=True)

_CV_SIGNALS = re.compile(
    r"(curriculum\s+vitae|c\.?\s*v\b|qualification|educational\s+background|"
    r"professional\s+experience|years\s+of\s+experience|b\.?\s*tech|m\.?\s*tech|"
    r"b\.?\s*e\b|m\.?\s*e\b|mba|msc|phd|ph\.d|post.?grad|graduate|"
    r"degree\s+in|diploma\s+in|certified|certification|"
    r"worked\s+(?:with|at|for|as)|employed\s+at|designation|"
    r"no\.\s+of\s+years|total\s+experience|relevant\s+experience)",
    re.IGNORECASE,
)

_ROLE_SYNONYMS: dict[str, list[str]] = {
    "team leader":    ["team leader", "project leader", "lead consultant", "team lead"],
    "procurement":    ["procurement", "purchase", "supply chain", "sourcing"],
    "gis":            ["gis", "geographic information", "remote sensing", "mapping"],
}


def _build_role_keywords(parameter: str) -> list[str]:
    param_lower = parameter.lower()
    kws = set()
    words = re.findall(r'\b[a-z]{4,}\b', param_lower)
    kws.update(w for w in words if w not in {
        "expert", "officer", "manager", "consultant", "senior", "junior",
        "lead", "head", "chief", "the", "and", "for", "with"
    })
    for role_key, synonyms in _ROLE_SYNONYMS.items():
        if role_key in param_lower:
            kws.update(synonyms)
    return list(kws)[:10]


def detect_cv_for_role(proposal_path: str, parameter: str, max_pages: int = 400) -> dict:
    if not _FITZ_OK or not Path(proposal_path).exists():
        return {"present": False, "confidence": "low", "evidence_page": None, "evidence_snippet": ""}
    keywords = _build_role_keywords(parameter)
    if not keywords:
        return {"present": False, "confidence": "low", "evidence_page": None, "evidence_snippet": ""}
    try:
        doc = fitz.open(proposal_path)
        total = min(len(doc), max_pages)
        best_page = None
        best_score = 0
        best_text = ""
        for pg_idx in range(total):
            txt = doc[pg_idx].get_text()
            low = txt.lower()
            kw_hits = sum(1 for kw in keywords if kw in low)
            if kw_hits == 0:
                continue
            cv_hits = len(_CV_SIGNALS.findall(txt))
            score = kw_hits * 2 + cv_hits
            if score > best_score:
                best_score = score
                best_page = pg_idx + 1
                for kw in keywords:
                    pos = low.find(kw)
                    if pos >= 0:
                        start = max(0, pos - 100)
                        end = min(len(txt), pos + 300)
                        best_text = txt[start:end].strip()
                        break
        doc.close()
        if best_score == 0:
            return {"present": False, "confidence": "low", "evidence_page": None, "evidence_snippet": ""}
        if best_score >= 6:    confidence = "high"
        elif best_score >= 3:  confidence = "medium"
        else:                  confidence = "low"
        return {
            "present":          best_score >= 2,
            "confidence":       confidence,
            "evidence_page":    best_page,
            "evidence_snippet": best_text[:200],
        }
    except Exception as e:
        print(f"[Analyzer] CV detect error for {parameter}: {e}")
        return {"present": False, "confidence": "low", "evidence_page": None, "evidence_snippet": ""}


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "have", "been",
    "each", "into", "over", "will", "are", "not", "its", "per", "any",
})


def _get_top_pages_for_criteria(
    proposal_path: str,
    criteria: list[dict],
    max_total_chars: int = 12_000,
) -> tuple[str, list[int]]:
    if not _FITZ_OK or not Path(proposal_path).exists():
        return "", []
    all_kws: set[str] = set()
    for c in criteria:
        if (c.get("formula_type") or "").upper() in ("QUAL", "LLM"):
            continue
        kws = c.get("search_keywords") or []
        for kw in kws:
            all_kws.update(re.findall(r'\b[a-z]{3,}\b', kw.lower()))
    all_kws -= _STOPWORDS
    if not all_kws:
        return "", []
    try:
        doc = fitz.open(proposal_path)
        scored = []
        for pg_idx in range(len(doc)):
            txt = doc[pg_idx].get_text()
            low = txt.lower()
            hits = sum(1 for kw in all_kws if kw in low)
            if hits > 0:
                scored.append((hits, pg_idx + 1, txt))
        doc.close()
    except Exception as e:
        print(f"[Analyzer] Page scan error: {e}")
        return "", []
    if not scored:
        return "", []
    scored.sort(reverse=True)
    parts: list[str] = []
    page_nos: list[int] = []
    total = 0
    for _, pno, txt in scored[:8]:
        block = f"[Page {pno}]\n{txt.strip()}"
        if total + len(block) > max_total_chars:
            block = block[:max_total_chars - total]
        parts.append(block)
        page_nos.append(pno)
        total += len(block)
        if total >= max_total_chars:
            break
    return "\n\n---\n\n".join(parts), page_nos


try:
    from core.proposal_analyzer_prompt_patch import BULK_EXTRACT_PROMPT as _BULK_EXTRACT_PROMPT
except ImportError:
    _BULK_EXTRACT_PROMPT = """Extract values.\n{criteria_list}\n{proposal_text}"""


def bulk_extract_values(
    proposal_path: str,
    criteria: list[dict],
) -> dict[str, dict]:
    from core.llm_client import call_llm, extract_json

    band_criteria = [
        c for c in criteria
        if not c.get("is_parent")
        and (c.get("formula_type") or "").upper() not in ("QUAL",)
    ]

    if not band_criteria:
        return {}

    criteria_list = "\n".join(
        f"- {c['parameter']} ({c.get('formula_type','?')}, max {c.get('max_marks',0)} marks): "
        f"{c.get('criteria_text','')[:150]}"
        for c in band_criteria
    )

    pages_text, _ = _get_top_pages_for_criteria(proposal_path, band_criteria)

    if not pages_text:
        print("[Analyzer] No proposal pages found for bulk extraction")
        return {}

    prompt = _BULK_EXTRACT_PROMPT.format(
        criteria_list=criteria_list,
        proposal_text=pages_text,
    )

    print(f"[Analyzer] Bulk extraction: {len(band_criteria)} criteria, "
          f"prompt={len(prompt):,} chars → 1 LLM call")

    raw = call_llm(prompt, label="bulk-extract")
    parsed = extract_json(raw) if raw else None

    if not parsed or not parsed.get("extractions"):
        print("[Analyzer] Bulk extraction: LLM returned no results")
        return {}

    return _remap_bulk_extractions(parsed.get("extractions", []), band_criteria)


def _hash_proposal(proposal_path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(proposal_path, "rb") as f:
            h.update(f.read(512 * 1024))
    except OSError:
        return ""
    return h.hexdigest()[:16]


def _load_proposal_cache(proposal_path: str, criteria_hash: str) -> Optional[dict]:
    h = _hash_proposal(proposal_path)
    if not h:
        return None
    cp = _CACHE_DIR / f"{h}.json"
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        if data.get("criteria_hash") != criteria_hash:
            return None
        return data
    except Exception:
        return None


def _save_proposal_cache(proposal_path: str, criteria_hash: str, data: dict) -> None:
    h = _hash_proposal(proposal_path)
    if not h:
        return
    cp = _CACHE_DIR / f"{h}.json"
    try:
        data["criteria_hash"] = criteria_hash
        data["cached_at"] = datetime.now().isoformat(timespec="seconds")
        cp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[Analyzer] Cache write error: {e}")


class ProposalAnalysis:
    def __init__(self, values: dict, cv_results: dict):
        self._values     = values
        self._cv_results = cv_results

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
        return hit.get("page") if hit else None

    def get_cv(self, parameter: str) -> dict:
        return self._cv_results.get(parameter, {
            "present": False, "confidence": "low",
            "evidence_page": None, "evidence_snippet": ""
        })

    def is_cv_present(self, parameter: str) -> bool:
        return bool(self.get_cv(parameter).get("present", False))

    def to_dict(self) -> dict:
        return {"values": self._values, "cv_results": self._cv_results}


def analyze_proposal(
    proposal_path: str,
    criteria: list[dict],
    force_refresh: bool = False,
) -> ProposalAnalysis:
    criteria_hash = hashlib.md5(
        json.dumps(sorted(c.get("parameter","") for c in criteria)).encode()
    ).hexdigest()[:8]

    if not force_refresh:
        cached = _load_proposal_cache(proposal_path, criteria_hash)
        if cached:
            return ProposalAnalysis(
                values=cached.get("values", {}),
                cv_results=cached.get("cv_results", {}),
            )

    print(f"[Analyzer] Analyzing proposal: {Path(proposal_path).name}")
    values = bulk_extract_values(proposal_path, criteria)

    cv_results: dict = {}
    qual_criteria = [
        c for c in criteria
        if not c.get("is_parent")
        and (c.get("formula_type") or "").upper() in ("QUAL", "LLM")
    ]
    if qual_criteria:
        for c in qual_criteria:
            parameter = c.get("parameter", "")
            result = detect_cv_for_role(proposal_path, parameter)
            cv_results[parameter] = result

    analysis = ProposalAnalysis(values=values, cv_results=cv_results)
    _save_proposal_cache(proposal_path, criteria_hash, analysis.to_dict())
    return analysis