"""
core/tq_criteria_extractor.py
==============================
Stage 1 — Extract the technical evaluation marking scheme from any RFP PDF.

Strategy (waterfall — each stage is tried in order, first success wins):

  A. TOC scan  →  detect the eval section page range
  B. Page scoring  →  rank every page by signal density
  C. Text-line reconstruction  →  parse rows from reading-order text
  D. Ollama LLM fallback  →  send raw page text, ask for JSON table

Output: list of criterion dicts
  {item_code, parameter, max_marks, criteria_text, formula_hint}

formula_hint is pre-tagged here so the scorer doesn't have to re-detect:
  STEP | BAND | PER_UNIT | QUAL | BINARY | LLM
"""

from __future__ import annotations

import re
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_MODEL    = "llama3.2"
OLLAMA_TIMEOUT  = 300        # 5 min — table extraction can be slow
MAX_CONTEXT_CHARS = 30_000
MAX_SECTION_SPAN  = 25       # max pages an eval section can span
SNO_MAX           = 25       # highest S.No we'll believe

# ──────────────────────────────────────────────────────────────────────────────
# Skip / sub-item patterns
# ──────────────────────────────────────────────────────────────────────────────

_SKIP = re.compile(
    r"""(
        \bpresentation\b | \binterview\b | \bviva\b | \bdemo\b |
        financial\s+bid  | price\s+bid   | \bL1\b  |
        commercial\s+bid | combined\s+and\s+final |
        indemnity        | arbitration   |
        opening\s+of.*financial | evaluation\s+of\s+financial
    )""",
    re.I | re.X,
)

_SUBITEM = re.compile(
    r"""^(
        team\s+leader  | procurement\s+expert | documentation\s+expert |
        urban\s+plann  | environmental\s+expert | ict\s*/\s*it |
        gis\s+expert   | data\s+analyst | legal\s+policy |
        finance\s+expert | reporting\s+manager | liaison\s+officer |
        ppp\s+specialist | social\s+development | monitoring\s+expert
    )""",
    re.I | re.X,
)

_TOR_ACTION = re.compile(
    r"""^(
        assist\b | monitor\b | submit\b | prepare\b | coordinate\b |
        ensure\b | must\s+be | the\s+consultant\s+shall
    )""",
    re.I | re.X,
)

# ──────────────────────────────────────────────────────────────────────────────
# Signal regexes for page scoring
# ──────────────────────────────────────────────────────────────────────────────

_MARKS_SIG = re.compile(
    r"(\d+\s*marks?\b|max(?:imum)?\.?\s*marks?|\d+\s*points?)",
    re.I,
)
_PARAM_SIG = re.compile(
    r"(turnover|experience|qualification|competence|methodology|"
    r"personnel|manpower|professional|net\s*worth|revenue|certification|"
    r"average\s+annual|work\s+order|project|assignment)",
    re.I,
)
_TABLE_HEADER = re.compile(
    r"s[\.\s]*no\.?.{0,200}(parameter|criterion|particulars).{0,200}"
    r"(max(?:imum)?\.?\s*marks?|full\s+marks?)",
    re.I | re.DOTALL,
)
_EVAL_KW = re.compile(
    r"(criteria\s+for\s+(technical\s+)?evaluation|evaluation\s+(of\s+)?criteria|"
    r"evaluation\s+of\s+technical\s+bid|technical\s+bid\s+eval|scoring\s+criteria|"
    r"marking\s+scheme|evaluation\s+matrix)",
    re.I,
)
_NEXT_KW = re.compile(
    r"(short.?list|evaluation\s+of\s+financial|financial\s+bid\s+eval|"
    r"combined\s+and\s+final|general\s+conditions|fraud\s+and\s+corrupt|"
    r"special\s+conditions\s+of\s+contract)",
    re.I,
)


# ──────────────────────────────────────────────────────────────────────────────
# Ollama helper
# ──────────────────────────────────────────────────────────────────────────────

def _ollama(prompt: str, timeout: int = OLLAMA_TIMEOUT) -> str:
    import requests
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 8192},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"] or ""
    except Exception as e:
        print(f"[CriteriaExtractor] Ollama error: {e}")
        return ""


def _parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:            esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"':      in_str = not in_str; continue
        if in_str:         continue
        if ch == "{":      depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = re.sub(r",\s*([}\]])", r"\1", text[start:i+1])
                try: return json.loads(candidate)
                except: return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# A. TOC scan
# ──────────────────────────────────────────────────────────────────────────────

def _trailing_int(line: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3})\s*$", line.rstrip())
    return int(m.group(1)) if m else None


def _toc_range(doc: fitz.Document) -> tuple[Optional[int], Optional[int]]:
    toc_text = "".join(doc[i].get_text() for i in range(min(20, len(doc))))
    lines = toc_text.splitlines()
    start_pg = end_pg = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if start_pg is None and _EVAL_KW.search(line):
            pg = _trailing_int(line)
            if pg and 5 <= pg <= 250:
                start_pg = pg
                for j in range(i + 1, min(i + 20, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        continue
                    if _NEXT_KW.search(nxt):
                        ep = _trailing_int(nxt)
                        if ep and ep >= start_pg:
                            end_pg = ep
                        break
                    ep = _trailing_int(nxt)
                    if ep and ep > start_pg:
                        end_pg = ep
                        break
                break
    if start_pg:
        print(f"[CriteriaExtractor] TOC: eval section p{start_pg}→p{end_pg}")
    return start_pg, end_pg


# ──────────────────────────────────────────────────────────────────────────────
# B. Page scoring
# ──────────────────────────────────────────────────────────────────────────────

def _score_pages(doc: fitz.Document, toc_start: Optional[int],
                 toc_end: Optional[int]) -> dict[int, float]:
    scores: dict[int, float] = {}
    hot: set[int] = set()

    toc_lo = (toc_start - 1) if toc_start else None
    toc_hi = (min(toc_end, toc_start + MAX_SECTION_SPAN) + 1
               if toc_start and toc_end else
               toc_start + MAX_SECTION_SPAN + 1 if toc_start else None)

    for pno in range(len(doc)):
        page_num = pno + 1
        txt = doc[pno].get_text()
        sc = 0.0

        if toc_lo is not None and toc_hi is not None and toc_lo <= page_num <= toc_hi:
            sc += 10
        if _TABLE_HEADER.search(txt):
            sc += 9
        mhits = len(_MARKS_SIG.findall(txt))
        sc += min(mhits * 1.5, 8)
        phits = len(_PARAM_SIG.findall(txt))
        sc += min(phits * 0.8, 5)
        if _EVAL_KW.search(txt):
            sc += 4

        scores[page_num] = sc
        if sc >= 8:
            hot.add(page_num)

    # proximity bonus
    for pno, sc in list(scores.items()):
        bonus = sum(2 for adj in [pno-1, pno+1] if adj in hot)
        scores[pno] = sc + bonus

    return scores


def _best_cluster(scores: dict[int, float], toc_start: Optional[int]) -> list[int]:
    selected = sorted(p for p, s in scores.items() if s >= 6)
    if not selected:
        return []
    clusters: list[list[int]] = [[selected[0]]]
    for pg in selected[1:]:
        if pg - clusters[-1][-1] <= 2:
            clusters[-1].append(pg)
        else:
            clusters.append([pg])

    def _weight(c):
        s = [scores[p] for p in c]
        return len(c) * (sum(s) / len(s))

    if toc_start is not None:
        top2 = sorted(clusters, key=_weight, reverse=True)[:2]
        best = min(top2, key=lambda c: min(abs(p - toc_start) for p in c))
    else:
        best = max(clusters, key=_weight)

    if best[-1] - best[0] > MAX_SECTION_SPAN:
        best = [p for p in best if p <= best[0] + MAX_SECTION_SPAN]
    return best


# ──────────────────────────────────────────────────────────────────────────────
# C. Text-line reconstruction (deterministic, no LLM)
# ──────────────────────────────────────────────────────────────────────────────

_SNO_RE = re.compile(r"^\s*(\d{1,2})[.\)]\s+\S")
_NUM_RE = re.compile(r"\b(\d{1,3})\b")


def _find_max_marks(text: str) -> Optional[int]:
    """Find max marks integer (5–60) in a block; avoid years and crore amounts."""
    # Prefer "N marks" patterns
    m = re.search(r"(\d+)\s*marks?\b", text, re.I)
    if m:
        v = int(m.group(1))
        if 3 <= v <= 60:
            return v
    # Fallback: largest integer 3–60 not followed by Cr/% 
    candidates = []
    for mm in _NUM_RE.finditer(text):
        v = int(mm.group(1))
        if 3 <= v <= 60:
            after = text[mm.end():mm.end()+5].lower()
            if re.match(r"\s*(?:cr|lakh|%|rs)", after):
                continue
            if 2000 <= v <= 2030:   # year
                continue
            candidates.append(v)
    return max(candidates) if candidates else None


def _textline_extract(doc: fitz.Document, cluster: list[int]) -> list[dict]:
    """Parse scoring table rows from reading-order text lines."""
    lines: list[tuple[int, str]] = []
    for pg in cluster:
        if pg < 1 or pg > len(doc):
            continue
        for line in doc[pg-1].get_text("text").splitlines():
            lines.append((pg, line))

    rows = []
    i = 0
    while i < len(lines):
        pg, line = lines[i]
        m = _SNO_RE.match(line)
        if m:
            sno = int(m.group(1))
            if 1 <= sno <= SNO_MAX:
                # collect block until next S.No
                block = [line.strip()]
                j = i + 1
                while j < len(lines) and j < i + 40:
                    _, nxt = lines[j]
                    nm = _SNO_RE.match(nxt)
                    if nm and int(nm.group(1)) != sno and int(nm.group(1)) <= SNO_MAX:
                        break
                    block.append(nxt.strip())
                    j += 1

                block_text = " ".join(block)
                max_marks = _find_max_marks(block_text)
                if max_marks:
                    # parameter: first meaningful text after S.No prefix
                    param_raw = re.sub(r"^\s*\d{1,2}[.\)]\s+", "", line).strip()
                    param = re.split(r"\s{3,}", param_raw)[0][:100].strip()
                    # criteria_text: everything in block
                    crit = re.sub(r"^\s*\d{1,2}[.\)]\s+", "", block_text)
                    crit = re.sub(r"\b\d+\s*marks?\b", "", crit, flags=re.I)
                    crit = " ".join(crit.split()).strip()[:1200]

                    rows.append({
                        "item_code": str(sno),
                        "parameter": param,
                        "max_marks": max_marks,
                        "criteria_text": crit,
                    })
                i = j
                continue
        i += 1
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# D. LLM fallback
# ──────────────────────────────────────────────────────────────────────────────

_LLM_TABLE_PROMPT = """\
You are reading pages from an Indian government RFP/tender document.
Extract the Technical Bid Evaluation scoring table.

RULES — read carefully:
1. ONE row per S.No (1, 2, 3 ...). Count visible S.No values and produce exactly that many rows.
2. max_marks = the integer in the "Max. Marks" / "Maximum Marks" column ONLY.
   Do NOT derive marks from band text inside criteria column.
3. parameter = the SHORT label in the second column (10-60 chars).
4. criteria_text = the FULL verbatim text from the "Particulars / Criteria" column.
5. SKIP rows about: Presentation, Financial Bid, Interview, Price Bid, Contract terms.
6. If qualification experts are listed as sub-items inside one S.No, keep them in criteria_text.

RFP PAGES:
{context}

Return ONLY valid JSON:
{{
  "grand_total_marks": <integer or null>,
  "qualification_threshold_pct": <number or null>,
  "criteria": [
    {{
      "item_code": "<S.No>",
      "parameter": "<short label>",
      "max_marks": <integer>,
      "criteria_text": "<full verbatim criteria>"
    }}
  ]
}}"""


def _llm_extract(cluster: list[int], doc: fitz.Document) -> list[dict]:
    parts, total = [], 0
    for pg in cluster:
        if pg < 1 or pg > len(doc):
            continue
        block = f"[Page {pg}]\n{doc[pg-1].get_text().strip()}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            block = block[:MAX_CONTEXT_CHARS - total]
        parts.append(block)
        total += len(block)
        if total >= MAX_CONTEXT_CHARS:
            break

    prompt = _LLM_TABLE_PROMPT.format(context="\n\n---\n\n".join(parts))
    print(f"[CriteriaExtractor] LLM fallback: {len(prompt)} chars prompt")
    raw = _ollama(prompt)
    if not raw:
        return []
    data = _parse_json(raw)
    if not data or "criteria" not in data:
        return []
    return [c for c in data["criteria"]
            if c.get("parameter") and int(c.get("max_marks") or 0) > 0]


# ──────────────────────────────────────────────────────────────────────────────
# Validation & deduplication
# ──────────────────────────────────────────────────────────────────────────────

def _validate(rows: list[dict]) -> list[dict]:
    valid = []
    seen: dict = {}
    for c in rows:
        p  = (c.get("parameter") or "").strip()
        mm = int(c.get("max_marks") or 0)
        if not p or mm < 2 or mm > 80:
            continue
        if _SKIP.search(p):
            print(f"[CriteriaExtractor] skip: {p[:60]}")
            continue
        if _SUBITEM.match(p):
            print(f"[CriteriaExtractor] sub-item: {p[:60]}")
            continue
        if _TOR_ACTION.match(p):
            print(f"[CriteriaExtractor] tor-action: {p[:60]}")
            continue
        key = (re.sub(r"\s+", " ", p).lower(), mm)
        ct = (c.get("criteria_text") or "")
        if key not in seen or len(ct) > len(seen[key].get("criteria_text", "")):
            seen[key] = {**c, "parameter": p, "max_marks": mm}
    return list(seen.values())


# ──────────────────────────────────────────────────────────────────────────────
# Formula hint tagging
# ──────────────────────────────────────────────────────────────────────────────

def _tag_formula(parameter: str, criteria_text: str) -> str:
    ct  = criteria_text.lower()
    p   = parameter.lower()

    # STEP: base turnover + increments per additional Cr
    if re.search(r"every\s+additional\s+\d+\s*cr", ct, re.I):
        return "STEP"
    if re.search(r"for\s+turnover.*?more\s+than|above\s+\d+\s*cr.*?\d+\s*marks?", ct, re.I):
        return "STEP"

    # BAND: explicit threshold bands for professionals/manpower
    if len(re.findall(r"\d+\s*professional", ct, re.I)) >= 2:
        return "BAND"
    if len(re.findall(r"\d+\s*employees?", ct, re.I)) >= 2:
        return "BAND"
    if re.search(r"(up\s+to|more\s+than)\s+\d+\s*(cr|crore|lakh)", ct, re.I) and \
       len(re.findall(r"\d+\s*marks?", ct, re.I)) >= 2:
        return "BAND"

    # PER_UNIT: N marks per project/assignment, capped
    if re.search(r"\d+\s*marks?\s+for\s+(each|01|per|one)\s+(project|assignment)", ct, re.I):
        return "PER_UNIT"
    if re.search(r"(each|per)\s+(project|assignment).*?\d+\s*marks?", ct, re.I):
        return "PER_UNIT"

    # QUAL: qualification/competence check — CV-based
    if re.search(r"(qualification|competence|cv|curriculum\s+vitae|"
                 r"post\s+graduate|years\s+of\s+exp)", ct, re.I):
        return "QUAL"
    if re.search(r"(qualification|competence)", p, re.I):
        return "QUAL"

    # BINARY: yes/no criteria (registration, certification, etc.)
    if re.search(r"(registered|certified|accredited|empanelled|"
                 r"iso\s+\d+|startup\s+india|msme)", ct, re.I):
        return "BINARY"

    # Default: LLM will evaluate
    return "LLM"


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def extract_marking_scheme(pdf_path: str) -> dict:
    """
    Extract the technical evaluation marking scheme from an RFP PDF.

    Returns:
    {
        "criteria": list[{item_code, parameter, max_marks, criteria_text, formula_hint}],
        "grand_total_marks": int,
        "qualification_threshold_pct": float,
        "eval_pages": [list of page numbers],
        "extraction_method": str,
        "schema_warning": str | None,
        "error": str | None,
    }
    """
    if not Path(pdf_path).exists():
        return _err(f"File not found: {pdf_path}")

    print(f"[CriteriaExtractor] Processing: {Path(pdf_path).name}")
    doc = fitz.open(str(pdf_path))

    try:
        # Stage A/B: find page cluster
        toc_start, toc_end = _toc_range(doc)
        scores = _score_pages(doc, toc_start, toc_end)
        cluster = _best_cluster(scores, toc_start)

        if not cluster:
            # Fallback: use TOC range or pages 40-55
            lo = max(1, (toc_start or 40) - 2)
            hi = min(len(doc), (toc_end or lo + 10) + 2)
            cluster = list(range(lo, hi + 1))
            print(f"[CriteriaExtractor] No cluster — using p{lo}–p{hi}")
        else:
            print(f"[CriteriaExtractor] Cluster: p{cluster[0]}–p{cluster[-1]}")

        # Stage C: text-line extraction
        rows = _textline_extract(doc, cluster)
        method = "textline"

        # Stage D: LLM fallback if text-line got nothing useful
        if len(rows) < 3 or sum(c["max_marks"] for c in rows) < 15:
            print(f"[CriteriaExtractor] Text-line insufficient "
                  f"({len(rows)} rows) — using LLM fallback")
            llm_rows = _llm_extract(cluster, doc)
            if len(llm_rows) >= len(rows):
                rows = llm_rows
                method = "llm"

        valid = _validate(rows)
        if not valid:
            doc.close()
            return _err("No valid criteria extracted after all strategies.")

        # Tag formula hints
        for c in valid:
            c["formula_hint"] = _tag_formula(c["parameter"], c.get("criteria_text", ""))

        doc_max = sum(c["max_marks"] for c in valid)
        warn = None
        if doc_max < 20:
            warn = f"Only {doc_max} total marks extracted — likely missed rows."
        if len(valid) < 3:
            warn = f"Only {len(valid)} criteria found — table extraction may be incomplete."

        print(f"[CriteriaExtractor] {len(valid)} criteria | "
              f"total={doc_max} | method={method}")
        for c in valid:
            print(f"  [{c['item_code']:>2}] {c['parameter'][:55]:55s} "
                  f"{c['max_marks']:>3} marks  [{c['formula_hint']}]")

        doc.close()
        return {
            "criteria":                    valid,
            "grand_total_marks":           doc_max,
            "qualification_threshold_pct": 70.0,
            "eval_pages":                  cluster,
            "extraction_method":           method,
            "schema_warning":              warn,
            "error":                       None,
        }

    except Exception as e:
        try: doc.close()
        except Exception: pass
        return _err(str(e))


def _err(msg: str) -> dict:
    print(f"[CriteriaExtractor] ERROR: {msg}")
    return {
        "criteria": [], "grand_total_marks": 0,
        "qualification_threshold_pct": 70.0,
        "eval_pages": [], "extraction_method": "none",
        "schema_warning": None, "error": msg,
    }
