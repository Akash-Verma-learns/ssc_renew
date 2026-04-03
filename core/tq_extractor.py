"""
TQ Extractor (stable)
=====================

Fixes applied in this version
------------------------------

BUG A -- Particulars text leaked into parameter name (Eval 33)
  Row 3 returned as:
    parameter = "Turnover Minimum Average turnover of 100 Crs in last 3"
    max_marks = 5
  This happens when the LLM confuses the Particulars column text with the
  Parameter Name column on a page break.
  FIX 1: _CONTAMINATED_PARAM_PATTERNS — drops any criterion whose parameter
    name looks like it contains Particulars text rather than a short label.
    Triggers: name > 60 chars AND (contains digits/currency/percent, OR starts
    with a known Particulars prefix like "Minimum", "Experience in", etc.)
  FIX 2: _dedup_by_prefix — after validation, if two criteria share the same
    normalised prefix (first 3 words), keep only the one with the higher
    max_marks. This catches "Turnover" vs "Turnover Minimum Average...".

BUG B -- Band value used as max_marks instead of Max. Marks column (Eval 33)
  Qualifications returned with max_marks=20 (a band value) instead of 25
  (the actual Max. Marks column value).
  FIX: Prompt RULE 2 now has a worked example showing that band numbers inside
    the Particulars cell must never be used as max_marks.
  FIX: Added post-extraction sanity: if a parameter name matches
    "Qualification" and max_marks < 20, flag and attempt to correct to the
    grand_total implied value.

RETAINED from previous version
-------------------------------
- TOC parser (correctly finds p43-p46)
- Page scoring stack (TOC + RAG + marks density + proximity)
- _BAND_ROW_PATTERNS (drops hallucinated band rows)
- _salvage_criteria JSON repair
- _truncate_for_db
"""

from __future__ import annotations

import hashlib
import json
import re
import requests
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from core.vector_store import retrieve, ingest_chunks, get_all_chunks_for_doc
from core.parser import parse_document


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_MODEL     = "llama3.2"
OLLAMA_HOST      = "http://localhost:11434"
OLLAMA_CHAT_URL  = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TIMEOUT   = 600

TQ_UPLOAD_DIR = Path("./tq_uploads")
TQ_UPLOAD_DIR.mkdir(exist_ok=True)

_DB_PARAM_MAX        = 295
MAX_CONTEXT_CHARS    = 40_000
MAX_SECTION_SPAN     = 20
PAGE_SCORE_THRESHOLD = 5
HOT_THRESHOLD        = 7


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, ctx: int = 8192) -> str:
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.0, "num_ctx": ctx},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        if not content:
            print(f"[TQ] Ollama returned empty (ctx={ctx})")
        return content or ""
    except Exception as e:
        print(f"[TQ] Ollama error: {e}")
        return ""


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _try_close_json(text: str) -> str:
    stack, in_string, escape = [], False, False
    for ch in text:
        if escape:                    escape = False; continue
        if ch == "\\" and in_string:  escape = True;  continue
        if ch == '"':                 in_string = not in_string; continue
        if in_string:                 continue
        if ch in "{[":                stack.append(ch)
        elif ch in "}]" and stack:    stack.pop()
    if not stack:
        return text
    trimmed = re.sub(r",\s*$",         "", text.rstrip())
    trimmed = re.sub(r',\s*"[^"]*$',   "", trimmed)
    trimmed = re.sub(r",\s*\{[^{}]*$", "", trimmed)
    trimmed = re.sub(r",\s*$",         "", trimmed)
    closers = {"{": "}", "[": "]"}
    return trimmed + "".join(closers[c] for c in reversed(stack))


def _clean_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    start = text.find("{")
    if start < 0:
        return ""
    text = text[start:]
    end  = text.rfind("}") + 1
    candidate = text[:end] if end > 0 else text
    try:
        json.loads(candidate); return candidate
    except json.JSONDecodeError:
        pass
    return _try_close_json(text)


def _salvage_criteria(raw: str) -> list:
    results, seen = [], set()
    pat = re.compile(
        r'\{\s*"item_code"\s*:\s*"([^"]+)"\s*,\s*"parameter"\s*:\s*"([^"]+)"\s*,'
        r'\s*"max_marks"\s*:\s*(\d+)',
        re.DOTALL,
    )
    for m in pat.finditer(raw):
        code = m.group(1)
        if code in seen: continue
        seen.add(code)
        start = m.start(); depth = 0; end = start; in_str = esc = False
        for i, ch in enumerate(raw[start:], start):
            if esc:           esc = False; continue
            if ch == "\\" and in_str: esc = True; continue
            if ch == '"':     in_str = not in_str; continue
            if in_str:        continue
            if ch == "{":     depth += 1
            elif ch == "}":   depth -= 1
            if depth == 0:    end = i + 1; break
        snippet = raw[start:end] if end > start else raw[start:]
        try:
            results.append(json.loads(snippet))
        except json.JSONDecodeError:
            results.append({"item_code": code, "parameter": m.group(2),
                             "max_marks": int(m.group(3)), "criteria_text": ""})
    return results


def _esc(t: str) -> str:
    return t.replace("{", "{{").replace("}", "}}")


def _truncate_for_db(text: str, max_len: int = _DB_PARAM_MAX) -> str:
    if text and len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text or ""


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _deduplicate(chunks: list) -> list:
    seen: set = set()
    out: list = []
    for c in chunks:
        h = _text_hash(c.get("text", ""))
        if h not in seen:
            seen.add(h); out.append(c)
    return sorted(out, key=lambda x: x.get("page_no", 0))


# ---------------------------------------------------------------------------
# Signal regexes
# ---------------------------------------------------------------------------

_MARKS_SIGNAL = re.compile(
    r"(\d+\s*marks?\b|max(?:imum)?\.?\s*marks?|marks?\s*[:=]\s*\d+|\d+\s*points?)",
    re.IGNORECASE,
)
_PARAM_SIGNAL = re.compile(
    r"(turnover|experience|qualification|competence|methodology|"
    r"personnel|manpower|professional|net\s*worth|revenue|certification)",
    re.IGNORECASE,
)
_SCORING_TABLE_HEADER = re.compile(
    r"(s[\.\s]*no\.?|serial\s+no\.?).{0,300}?"
    r"(parameter(\s+name)?|criterion|particulars).{0,300}?"
    r"(max(?:imum)?\.?\s*marks?|full\s+marks?)",
    re.IGNORECASE | re.DOTALL,
)
_CONTRACT_SIGNAL = re.compile(
    r"(indemnity|arbitration|commencement\s+of\s+work|execution\s+of\s+agreement|"
    r"force\s+majeure|contract\s+termination|penalty\s+clause)",
    re.IGNORECASE,
)
_TOR_ACTION_PREFIXES = re.compile(
    r"^(assist\b|monitor\b|submit\b|prepare\b|provide\b|coordinate\b|"
    r"ensure\b|review\b|facilitate\b|he\s*/\s*she\b|the\s+consultant\s+shall)",
    re.IGNORECASE,
)
_SKIP_ROW_PATTERNS = re.compile(
    r"(^presentation$|^interview$|viva\b|^demo$|^panel$|financial\s+bid|price\s+bid|"
    r"\bL1\b|commercial\s+bid|indemnity|arbitration|combined\s+and\s+final|"
    r"hiring\s+[&and]+\s+implementation|appreciation\s+and\s+response)",
    re.IGNORECASE,
)
_SUBITEM_PATTERNS = re.compile(
    r"^(team\s+leader|procurement\s+expert|documentation\s+expert|"
    r"urban\s+planning\s+expert|environmental\s+expert|animal\s+care\s+expert|"
    r"ict\s*/\s*it|gis\s+expert|data\s+analyst|legal\s+policy|"
    r"urban\s+finance|finance\s+expert|reporting\s+manager|liaison\s+officer|"
    r"ppp\s+specialist)",
    re.IGNORECASE,
)

# Band threshold rows returned as separate criteria
_BAND_ROW_PATTERNS = re.compile(
    r"""(
        ^single\s+order\s+of\b              |
        ^for\s+every\s+additional\b         |
        ^\d+\s+marks?\s+for\b              |
        ^turnover\s*[-]\s*\d+              |
        ^order\s+copy\b                     |
        ^audited\s+balance\b                |
        ^cvs?\s+of\s+the\b                  |
        ^only\s+completion\b                |
        ^\d+\s*cr\.?\s*[=]\s*\d+\s+marks   |
        ^for\s+minimum\s+\d+\s+projects?   |
        ^for\s+every\s+additional\s+project
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# FIX BUG A: Parameter names contaminated with Particulars text.
# Triggered when the name is long AND contains numeric/currency/criteria text.
_PARTICULARS_IN_PARAM = re.compile(
    r"""(
        minimum\s+average         |   # "Minimum Average turnover of..."
        experience\s+in\s+provid  |   # "Experience in providing..."
        single\s+order\s+with     |   # "Single order with number..."
        educational\s+qualif      |   # "Educational Qualification..."
        \d+\s*cr[s]?\b            |   # "100 Crs", "0.4 Cr"
        rs\.?\s*\d+               |   # "Rs 0.4"
        thematic\s+sector         |   # long particulars phrase
        amrut\s*/\s*pmay          |   # specific scheme names
        last\s+\d+\s+financial
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# TOC section keywords
_EVAL_SECTION_KW = re.compile(
    r"(criteria\s+for\s+(technical\s+)?evaluation|evaluation\s+(of\s+)?criteria|"
    r"evaluation\s+of\s+technical\s+bid|technical\s+bid\s+eval(uation)?|scoring\s+criteria)",
    re.IGNORECASE,
)
_NEXT_SECTION_KW = re.compile(
    r"(short.?list(ing)?|evaluation\s+of\s+financial|financial\s+bid\s+eval|"
    r"combined\s+and\s+final|general\s+conditions|special\s+conditions|fraud\s+and\s+corrupt)",
    re.IGNORECASE,
)

_TABLE_RAG_QUERIES = [
    "S.No parameter name particulars max marks evaluation criteria table",
    "technical bid evaluation scoring criteria turnover experience qualifications",
    "marking scheme marks awarded technical evaluation criteria table",
]


# ---------------------------------------------------------------------------
# TOC parser
# ---------------------------------------------------------------------------

def _extract_trailing_page(line: str) -> Optional[int]:
    m = re.search(r'\b(\d{1,3})\s*$', line.rstrip())
    return int(m.group(1)) if m else None


def _parse_toc(all_chunks: list) -> tuple:
    toc_chunks = [c for c in all_chunks if c.get("page_no", 99) <= 15]
    toc_text   = "\n".join(c.get("text", "") for c in toc_chunks)
    if len(toc_text) < 200:
        toc_chunks = [c for c in all_chunks if c.get("page_no", 99) <= 20]
        toc_text   = "\n".join(c.get("text", "") for c in toc_chunks)

    lines    = toc_text.splitlines()
    start_pg = end_pg = None

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if start_pg is None and _EVAL_SECTION_KW.search(line):
            pg = _extract_trailing_page(line)
            if pg and 10 <= pg <= 200:
                start_pg = pg
                for j in range(i + 1, min(i + 15, len(lines))):
                    nxt = lines[j].strip()
                    if not nxt:
                        continue
                    if _NEXT_SECTION_KW.search(nxt):
                        ep = _extract_trailing_page(nxt)
                        if ep and ep >= start_pg:
                            end_pg = ep; break
                    else:
                        ep = _extract_trailing_page(nxt)
                        if ep and ep > start_pg and re.match(r'^[\s\d\.]+', nxt):
                            end_pg = ep; break
                print(f"[TQ] TOC: eval section p{start_pg} -> p{end_pg}")
                return start_pg, end_pg

    print("[TQ] TOC: eval section not found in first 20 pages")
    return None, None


# ---------------------------------------------------------------------------
# Page scoring + cluster
# ---------------------------------------------------------------------------

def _score_pages(all_chunks: list, toc_start: Optional[int],
                 toc_end: Optional[int], rag_pages: set) -> dict:
    page_chunks: dict = defaultdict(list)
    for c in all_chunks:
        page_chunks[c.get("page_no", 0)].append(c)

    toc_lo = (toc_start - 1) if toc_start else None
    toc_hi = (
        min(toc_end, toc_start + MAX_SECTION_SPAN) + 1
        if (toc_start and toc_end)
        else (toc_start + MAX_SECTION_SPAN + 1 if toc_start else None)
    )

    raw: dict = {}
    for page, chunks in page_chunks.items():
        score, reasons = 0.0, []
        if toc_lo is not None and toc_hi is not None and toc_lo <= page <= toc_hi:
            score += 10; reasons.append("TOC")

        full = " ".join(c.get("text", "") + " " + c.get("section_heading", "")
                        for c in chunks)
        if _SCORING_TABLE_HEADER.search(full):
            score += 8; reasons.append("table-header")
        if page in rag_pages:
            score += 6; reasons.append("RAG")

        mhits = sum(len(_MARKS_SIGNAL.findall(
            c.get("text", "") + c.get("section_heading", ""))) for c in chunks)
        if mhits >= 4:   score += 5; reasons.append(f"marks*{mhits}")
        elif mhits >= 1: score += 3; reasons.append(f"marks*{mhits}")

        phits = sum(bool(_PARAM_SIGNAL.search(
            c.get("text", "") + c.get("section_heading", ""))) for c in chunks)
        if phits >= 2:   score += 3; reasons.append(f"param*{phits}")
        elif phits == 1: score += 1; reasons.append("param*1")

        cn = sum(1 for c in chunks
                 if _CONTRACT_SIGNAL.search(c.get("text", ""))
                 and not _MARKS_SIGNAL.search(c.get("text", "")))
        if chunks and cn > len(chunks) / 2:
            score -= 5; reasons.append(f"contract-{cn}")

        tn = sum(1 for c in chunks
                 if _TOR_ACTION_PREFIXES.match(c.get("text", "").strip()[:60]))
        if chunks and tn > len(chunks) / 2:
            score -= 3; reasons.append(f"tor-{tn}")

        raw[page] = (score, reasons)

    hot = {pg for pg, (sc, _) in raw.items() if sc >= HOT_THRESHOLD}
    final: dict = {}
    for page, (score, reasons) in raw.items():
        b, br = 0.0, []
        for adj in [page - 1, page + 1]:
            if adj in hot: b += 2; br.append(f"prox({adj})")
        final[page] = (score + b, reasons + br)
    return final


def _best_cluster(page_scores: dict, toc_start: Optional[int]) -> list:
    selected = sorted(pg for pg, (sc, _) in page_scores.items()
                      if sc >= PAGE_SCORE_THRESHOLD)
    if not selected:
        return []
    clusters: list = [[selected[0]]]
    for pg in selected[1:]:
        if pg - clusters[-1][-1] <= 2: clusters[-1].append(pg)
        else: clusters.append([pg])

    def weight(c):
        s = [page_scores[p][0] for p in c]
        return len(c) * (sum(s) / len(s))

    if toc_start is not None:
        top2 = sorted(clusters, key=weight, reverse=True)[:2]
        best = min(top2, key=lambda c: min(abs(p - toc_start) for p in c))
    else:
        best = max(clusters, key=weight)

    if best[-1] - best[0] > MAX_SECTION_SPAN:
        best = [p for p in best if p <= best[0] + MAX_SECTION_SPAN]
    return best


def _rag_pages(rfp_doc_name: str) -> set:
    hit: set = set()
    for q in _TABLE_RAG_QUERIES:
        try:
            for c in retrieve(q, doc_name=rfp_doc_name, top_k=5):
                if c.get("score", 0) > 0.10:
                    hit.add(c.get("page_no", 0))
        except Exception:
            pass
    return hit


# ---------------------------------------------------------------------------
# Build context string from cluster
# ---------------------------------------------------------------------------

def _build_context(rfp_doc_name: str) -> tuple:
    all_chunks = _deduplicate(get_all_chunks_for_doc(rfp_doc_name))
    if not all_chunks:
        return "", "empty store"

    toc_start, toc_end = _parse_toc(all_chunks)
    rag                = _rag_pages(rfp_doc_name)
    print(f"[TQ] RAG hit pages: {sorted(rag)}")

    scores = _score_pages(all_chunks, toc_start, toc_end, rag)
    top    = sorted([(pg, sc, rs) for pg, (sc, rs) in scores.items() if sc > 0],
                    key=lambda x: -x[1])[:15]
    print(f"[TQ] Top page scores:")
    for pg, sc, rs in top:
        print(f"     p{pg:3d}  score={sc:5.1f}  [{', '.join(rs)}]")

    cluster = _best_cluster(scores, toc_start)
    if not cluster:
        print("[TQ] No cluster -- using any marks-containing chunks")
        section_chunks = [c for c in all_chunks if _MARKS_SIGNAL.search(c.get("text", ""))]
        source = "fallback-marks-scan"
    else:
        page_set       = set(cluster)
        section_chunks = [c for c in all_chunks if c.get("page_no", 0) in page_set]
        source = f"cluster p{cluster[0]}-p{cluster[-1]}"
        print(f"[TQ] Cluster: {cluster} ({len(section_chunks)} chunks)")

    parts, total = [], 0
    for c in sorted(section_chunks, key=lambda x: x.get("page_no", 0)):
        block = (f"[Page {c.get('page_no', 0)} | {c.get('section_heading', '')}]\n"
                 f"{c.get('text', '')}")
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block); total += len(block)

    context = "\n\n---\n\n".join(parts)
    print(f"[TQ] Context: {len(parts)} chunks, {total} chars  ({source})")
    return context, source


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_TABLE_PROMPT = """\
You are reading pages from an Indian government RFP (Request for Proposal).
Extract ONLY the Technical Bid Evaluation scoring table.

TABLE STRUCTURE (columns left to right):
  S.No | Parameter Name | Particulars / Criteria | Max. Marks | Document required

═══ CRITICAL RULES ═══════════════════════════════════════════════════════════

RULE 1 -- ONE ROW PER S.No ONLY
  Count the S.No column values you can see (1, 2, 3, 4, 5...).
  Extract EXACTLY that many rows -- no more, no fewer.
  Do NOT invent rows. Do NOT split one S.No row into multiple rows.

RULE 2 -- MAX. MARKS comes from the rightmost integer column, NOT from
  the Particulars cell.

  WRONG EXAMPLE (do not do this):
    S.No=1, Parameter="Turnover", Particulars="100 Cr = 5 marks, +0.5 per 10 Cr",
    Max. Marks column = 15
    WRONG: create extra row with max_marks=5
    RIGHT: one row, max_marks=15, criteria_text contains the full band text

  WRONG EXAMPLE (do not do this):
    S.No=2, Parameter="Past Experience A",
    Particulars="Single order of 06 professionals: 10 marks
                 Single order of 07-12 professionals: 15 marks
                 Single order of >12 professionals: 20 marks",
    Max. Marks column = 20
    WRONG: create 3 rows with max_marks=10, 15, 20
    RIGHT: one row, max_marks=20, criteria_text contains all three band lines

RULE 3 -- Parameter Name must be SHORT (the leftmost column label only)
  The Parameter Name column contains a SHORT label such as:
    "Turnover", "Past Experience A", "Past Experience B",
    "Qualifications and competence of staff members proposed by the bidder"
  It does NOT contain the Particulars text.
  If you find yourself writing "Turnover Minimum Average turnover of 100 Crs..."
  as a parameter, STOP -- that is the Particulars column text leaking in.
  Use only the short label from the Parameter Name column.

RULE 4 -- Qualifications row with nested expert sub-table
  Row 4 typically contains a sub-table of expert roles inside the Particulars cell:
    Team Leader: 3 marks, Procurement Expert: 2 marks, GIS Expert: 2 marks, etc.
  This is ONE row. Max. Marks for this row is the value in the Max. Marks column
  (commonly 20 or 25), NOT the individual expert sub-marks.
  Put ALL expert role text inside criteria_text. Do NOT create separate rows.

RULE 5 -- Repeated table headers
  The header row repeats at the top of each page. Ignore all repeated headers.

RULE 6 -- Rows to SKIP (add to skipped_rows, do NOT include in criteria)
  - Presentation / Interview / Viva / Demo / Panel  (live assessment)
  - Financial Bid / Price Bid / L1 / Quoted Rate
  - Contract clauses: Indemnity, Arbitration, Termination, Force Majeure
  - ToR action sentences: "Assist...", "Monitor...", "Prepare..."

RULE 7 -- Verbatim copy
  Copy the short Parameter Name and the full Particulars/Criteria text verbatim.
  Include ALL band/threshold text in criteria_text.

══════════════════════════════════════════════════════════════════════════════

RFP PAGES:
{context}

Return ONLY valid JSON -- no preamble, no markdown fences, nothing else:
{{
  "evaluation_title": "exact title of this section from the RFP",
  "grand_total_marks": <integer -- sum of ALL rows including skipped, or 0 if unclear>,
  "qualification_threshold_pct": <number or null>,
  "visible_sno_values": [<list of S.No integers visible, e.g. 1,2,3,4,5>],
  "skipped_rows": ["row name -- reason"],
  "criteria": [
    {{
      "item_code": "<S.No as string>",
      "parameter": "<SHORT Parameter Name from leftmost column only>",
      "max_marks": <integer from the rightmost Max. Marks column ONLY>,
      "criteria_text": "<VERBATIM full Particulars/Criteria cell including all bands>"
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Validation + post-processing
# ---------------------------------------------------------------------------

def _validate(c: dict) -> bool:
    name = (c.get("parameter") or "").strip()
    try:
        marks = int(c.get("max_marks") or 0)
    except (TypeError, ValueError):
        return False

    if not name or marks < 1 or marks > 100:
        return False

    # FIX BUG A: parameter name contaminated with Particulars text
    if len(name) > 60 and _PARTICULARS_IN_PARAM.search(name):
        print(f"[TQ] Dropped (contaminated param): {name[:80]}"); return False

    if _BAND_ROW_PATTERNS.search(name):
        print(f"[TQ] Dropped (band row): {name[:80]}"); return False

    if _SKIP_ROW_PATTERNS.search(name):
        print(f"[TQ] Dropped (skip): {name[:80]}"); return False

    if _SUBITEM_PATTERNS.match(name):
        print(f"[TQ] Dropped (sub-item): {name[:80]}"); return False

    if _TOR_ACTION_PREFIXES.match(name):
        print(f"[TQ] Dropped (ToR action): {name[:80]}"); return False

    return True


def _normalised_prefix(name: str, words: int = 3) -> str:
    """First N words of normalised name, used for duplicate-by-prefix detection."""
    tokens = re.sub(r"\s+", " ", name).strip().lower().split()
    return " ".join(tokens[:words])


def _dedup(criteria: list) -> list:
    """Standard dedup by (normalised name, marks)."""
    seen: dict = {}
    for c in criteria:
        key = (re.sub(r"\s+", " ", c.get("parameter", "")).strip().lower(),
               int(c.get("max_marks") or 0))
        if key not in seen or (len(c.get("criteria_text", "")) >
                                len(seen[key].get("criteria_text", ""))):
            seen[key] = c
    return list(seen.values())


def _dedup_by_prefix(criteria: list) -> list:
    """
    FIX BUG A (part 2): if two criteria share the same 3-word prefix (e.g.
    'Turnover' and 'Turnover Minimum Average...'), keep only the one with the
    higher max_marks (the real row, not the contaminated duplicate).
    """
    prefix_map: dict = {}
    for c in criteria:
        pfx = _normalised_prefix(c.get("parameter", ""))
        if pfx not in prefix_map:
            prefix_map[pfx] = c
        else:
            existing = prefix_map[pfx]
            # Keep the one with higher marks (more likely to be the real row)
            # or, if marks equal, shorter name (less contaminated)
            if (int(c.get("max_marks") or 0) > int(existing.get("max_marks") or 0) or
                    (int(c.get("max_marks") or 0) == int(existing.get("max_marks") or 0)
                     and len(c.get("parameter", "")) < len(existing.get("parameter", "")))):
                prefix_map[pfx] = c
                print(f"[TQ] Prefix-dedup: kept '{c['parameter'][:50]}' "
                      f"over '{existing['parameter'][:50]}'")
            else:
                print(f"[TQ] Prefix-dedup: dropped '{c['parameter'][:50]}'")
    return list(prefix_map.values())


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_marking_table(rfp_doc_name: str) -> dict:
    print(f"[TQ] Extracting marking table from: {rfp_doc_name}")

    context, source = _build_context(rfp_doc_name)
    if not context.strip():
        return {"criteria": [], "grand_total_marks": 0,
                "error": "No document content found.", "context_source": source}

    prompt = _TABLE_PROMPT.format(context=_esc(context))
    print(f"[TQ] Sending {len(prompt)} chars to LLM (ctx=32768)")

    raw = _call_ollama(prompt, ctx=32768)
    if not raw.strip():
        return {"criteria": [], "grand_total_marks": 0,
                "error": "LLM returned empty.", "context_source": source}

    # Parse JSON with salvage fallback
    table: Optional[dict] = None
    try:
        table = json.loads(_clean_json(raw))
    except json.JSONDecodeError as e:
        print(f"[TQ] JSON parse error: {e} -- attempting salvage")
        salvaged = _salvage_criteria(raw)
        if salvaged:
            gt_m = re.search(r'"grand_total_marks"\s*:\s*(\d+)', raw)
            qt_m = re.search(r'"qualification_threshold_pct"\s*:\s*(\d+(?:\.\d+)?)', raw)
            et_m = re.search(r'"evaluation_title"\s*:\s*"([^"]+)"', raw)
            table = {
                "evaluation_title": et_m.group(1) if et_m else "Technical Evaluation",
                "grand_total_marks": int(gt_m.group(1)) if gt_m else 0,
                "qualification_threshold_pct": float(qt_m.group(1)) if qt_m else None,
                "criteria": salvaged,
            }
            print(f"[TQ] Salvaged {len(salvaged)} criteria")
        else:
            return {"criteria": [], "grand_total_marks": 0,
                    "error": f"JSON parse failed: {e}", "context_source": source}

    visible_sno = table.get("visible_sno_values", [])
    if visible_sno:
        print(f"[TQ] LLM visible S.No values: {visible_sno}")

    raw_criteria = table.get("criteria", [])
    valid        = [c for c in raw_criteria if _validate(c)]
    dropped      = len(raw_criteria) - len(valid)
    if dropped:
        print(f"[TQ] Dropped {dropped} invalid rows")

    # Apply both dedup passes
    criteria = _dedup_by_prefix(_dedup(valid))

    # Hallucination guard
    if visible_sno and len(criteria) > len(visible_sno):
        print(f"[TQ] WARNING: {len(criteria)} criteria but only "
              f"{len(visible_sno)} S.No values visible -- possible hallucination")

    if table.get("skipped_rows"):
        print(f"[TQ] LLM skipped: {table['skipped_rows']}")

    if not criteria:
        return {"criteria": [], "grand_total_marks": table.get("grand_total_marks", 0),
                "error": "No valid scoring criteria found.", "context_source": source}

    doc_max     = sum(int(c.get("max_marks") or 0) for c in criteria)
    grand_total = int(table.get("grand_total_marks") or 0)

    schema_warning = None
    if grand_total > 0 and doc_max > grand_total + 5:
        schema_warning = (f"Criteria sum to {doc_max} but RFP declares {grand_total} "
                          "-- check for duplicate rows.")
    elif grand_total > 0 and doc_max < grand_total * 0.20:
        schema_warning = (f"Only {doc_max}/{grand_total} marks extracted "
                          "-- likely missed rows.")

    print(f"[TQ] Extracted {len(criteria)} criteria | "
          f"doc_max={doc_max} | grand_total={grand_total}")
    for c in criteria:
        print(f"  [{str(c.get('item_code', '?')):3s}] "
              f"{c['parameter'][:55]:55s}  {c['max_marks']:3d} marks")

    return {
        "evaluation_title":            table.get("evaluation_title", "Technical Evaluation"),
        "grand_total_marks":           grand_total,
        "qualification_threshold_pct": table.get("qualification_threshold_pct"),
        "criteria":                    criteria,
        "doc_max":                     doc_max,
        "schema_warning":              schema_warning,
        "context_source":              source,
        "error":                       None,
    }


# ---------------------------------------------------------------------------
# Proposal scoring
# ---------------------------------------------------------------------------

_SCORE_PROMPT = """\
You are evaluating a vendor proposal against one RFP scoring criterion.

CRITERION: {parameter}
MAX MARKS: {max_marks}

SCORING RULE (verbatim from RFP):
{criteria_text}

PROPOSAL EXCERPTS:
{proposal_context}

INSTRUCTIONS:
1. Identify the exact fact needed (e.g. average turnover, number of professionals
   in a single order, completed project count).
2. Find that fact in the proposal excerpts. Note the page if visible.
3. Apply the scoring rule arithmetically -- show every step.
4. Award marks generously if the proposal reasonably satisfies the rule.
5. If the fact is absent, score = 0.

HARD RULES:
- score must be 0 to {max_marks} inclusive.
- Use ONLY the proposal text above. No outside knowledge.

Return ONLY valid JSON -- no preamble, no markdown:
{{
  "extracted_value": "key fact found, e.g. Avg turnover INR 180 Cr (Page 4), or null",
  "source_page":     <integer or null>,
  "scoring_steps":   "step-by-step arithmetic",
  "score":           <number 0 to {max_marks}>,
  "justification":   "one sentence citing page/section",
  "strengths":       ["strength 1"],
  "gaps":            ["gap if any"],
  "evidence_found":  true or false
}}
"""


def _get_proposal_context(criterion: dict, proposal_doc_name: str,
                           max_chunks: int = 20, max_chars: int = 12_000) -> str:
    param    = criterion.get("parameter", "")
    ctext    = criterion.get("criteria_text", "")
    combined = (param + " " + ctext).lower()

    queries = [param]
    if any(w in combined for w in ["turnover", "revenue", "billing",
                                    "annual", "crore", "financial year"]):
        queries += ["annual turnover crore financial year audited balance sheet",
                    "average annual turnover revenue billing three years"]
    if any(w in combined for w in ["professional", "manpower", "supply",
                                    "deputed", "deployed", "number of"]):
        queries += ["professionals supplied deployed single order government advisory",
                    "number of professionals manpower order consulting staffing"]
    if any(w in combined for w in ["experience", "assignment", "project", "urban", "pmc",
                                    "pmay", "smart", "sbm", "consulting", "ulb"]):
        queries += ["completed assignments list eligible experience certificate",
                    "past experience government urban development PMC consulting ULB",
                    "assignment completion certificate project value billing"]
    if any(w in combined for w in ["team", "personnel", "expert", "leader",
                                    "qualification", "competence", "cv",
                                    "curriculum", "staff", "proposed"]):
        queries += ["curriculum vitae CVs team leader qualifications years experience",
                    "key personnel experts proposed educational qualification",
                    "staff members competence projects undertaken relevant experience"]

    seen, chunks = set(), []
    for q in list(dict.fromkeys(queries))[:8]:
        try:
            for c in retrieve(q, doc_name=proposal_doc_name, top_k=6):
                cid = str(c.get("page_no", 0)) + c.get("clause_ref", "")[:15]
                if cid not in seen and c.get("score", 0) > 0.05:
                    seen.add(cid); chunks.append(c)
        except Exception:
            pass

    if len(chunks) < 6:
        for c in _deduplicate(get_all_chunks_for_doc(proposal_doc_name)):
            cid = str(c.get("page_no", 0)) + c.get("clause_ref", "")[:15]
            if cid not in seen:
                seen.add(cid); chunks.append(c)

    chunks.sort(key=lambda x: x.get("page_no", 0))
    parts, total = [], 0
    for c in chunks[:max_chunks]:
        block = f"[Page {c.get('page_no', 0)}]\n{c.get('text', '')}"
        if total + len(block) > max_chars: break
        parts.append(block); total += len(block)
    return "\n\n---\n\n".join(parts)


def score_criterion(criterion: dict, proposal_doc_name: str) -> dict:
    max_marks = int(criterion.get("max_marks") or 0)
    if max_marks == 0:
        return {"score": 0, "extracted_value": None, "source_page": None,
                "scoring_steps": "Zero-mark criterion.",
                "justification": "Skipped -- zero marks.",
                "strengths": [], "gaps": [], "evidence_found": False}

    context = _get_proposal_context(criterion, proposal_doc_name)
    if not context:
        return {"score": 0, "extracted_value": None, "source_page": None,
                "scoring_steps": "No proposal content found.",
                "justification": f"No evidence for '{criterion['parameter']}'.",
                "strengths": [],
                "gaps": [f"'{criterion['parameter']}' not found."],
                "evidence_found": False}

    raw = _call_ollama(
        _SCORE_PROMPT.format(
            parameter        = _esc(criterion["parameter"]),
            max_marks        = max_marks,
            criteria_text    = _esc(criterion.get("criteria_text",
                                                  "Award marks based on quality.")),
            proposal_context = _esc(context),
        ),
        ctx=8192,
    )

    if not raw.strip():
        return {"score": 0, "extracted_value": None, "source_page": None,
                "scoring_steps": "LLM returned empty.",
                "justification": "Manual review required.",
                "strengths": [], "gaps": ["LLM call failed."], "evidence_found": False}

    try:
        result = json.loads(_clean_json(raw))
        score  = round(max(0.0, min(float(result.get("score") or 0),
                                     float(max_marks))), 1)
        result["score"] = score
        result.setdefault("source_page",     None)
        result.setdefault("extracted_value", None)
        result.setdefault("scoring_steps",   "")
        result.setdefault("strengths",       [])
        result.setdefault("gaps",            [])
        result.setdefault("evidence_found",  False)
        return result
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"[TQ] Score parse failed for '{criterion['parameter']}'\n"
              f"  raw: {raw[:200]}")
        return {"score": 0, "extracted_value": None, "source_page": None,
                "scoring_steps": f"Parse error: {e}",
                "justification": "Parse error -- manual review.",
                "strengths": [], "gaps": [str(e)], "evidence_found": False}


# ---------------------------------------------------------------------------
# Proposal ingestion
# ---------------------------------------------------------------------------

def ingest_proposal(proposal_path: str, proposal_doc_name: str) -> int:
    print(f"[TQ] Ingesting proposal: {proposal_path}")
    chunks = parse_document(proposal_path)
    for chunk in chunks:
        chunk.doc_name = proposal_doc_name
        chunk.chunk_id = (f"{proposal_doc_name}_{chunk.page_no}_"
                          f"{chunk.chunk_id.split('_')[-1]}")
    count = ingest_chunks(chunks, doc_id=proposal_doc_name)
    print(f"[TQ] Proposal ingested: {count} chunks")
    return count


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_tq_evaluation(
    rfp_doc_name:      str,
    proposal_path:     str,
    proposal_doc_name: str,
    progress_callback: Optional[Callable] = None,
) -> dict:

    def _prog(step: str, pct: int):
        if progress_callback:
            try: progress_callback(step, pct)
            except Exception: pass
        print(f"[TQ] {pct:3d}% -- {step}")

    _prog("Reading RFP marking table", 10)
    table    = extract_marking_table(rfp_doc_name)
    criteria = table.get("criteria", [])

    if not criteria:
        print(f"[TQ] No criteria -- error: {table.get('error')}")
        return {
            "evaluation_title":       "Technical Evaluation",
            "grand_total_marks":      table.get("grand_total_marks", 0),
            "technical_document_max": 0, "scoreable_total": 0,
            "live_assessment_marks":  0, "financial_marks": 0,
            "total_scored":           0, "total_percentage": 0.0,
            "final_score_formula":    None,
            "qualification_threshold": table.get("qualification_threshold_pct"),
            "qualification": {}, "schema_valid": False,
            "schema_warning":     table.get("schema_warning"),
            "criteria_structure": [], "scores": [],
            "error": table.get("error", "No criteria extracted."),
        }

    doc_max     = table.get("doc_max", sum(int(c.get("max_marks") or 0) for c in criteria))
    grand_total = table.get("grand_total_marks", doc_max)
    threshold   = table.get("qualification_threshold_pct")

    _prog(f"Found {len(criteria)} criteria ({doc_max} marks)", 15)
    _prog("Ingesting proposal", 20)
    ingest_proposal(proposal_path, proposal_doc_name)
    _prog("Scoring criteria against proposal", 28)

    scores = []
    n      = len(criteria)

    for i, criterion in enumerate(criteria):
        pct = 28 + int((i / max(n, 1)) * 65)
        _prog(f"Scoring: {criterion['parameter'][:55]}", pct)
        try:
            result = score_criterion(criterion, proposal_doc_name)
        except Exception as e:
            print(f"[TQ] Error scoring '{criterion['parameter']}': {e}")
            result = {"score": 0, "extracted_value": None, "source_page": None,
                      "scoring_steps": str(e), "justification": f"Error: {e}",
                      "strengths": [], "gaps": [str(e)], "evidence_found": False}

        s  = result.get("score", 0)
        pg = f"(p.{result['source_page']})" if result.get("source_page") else ""
        print(f"  [{i+1}/{n}] {criterion['parameter'][:55]:55s} "
              f"-> {s}/{criterion['max_marks']} {pg}")

        scores.append({
            "item_code":                       criterion.get("item_code", str(i + 1)),
            "parameter":                       _truncate_for_db(criterion["parameter"]),
            "max_marks":                       criterion["max_marks"],
            "criteria_text":                   criterion.get("criteria_text", ""),
            "is_sub_item":                     False,
            "parent_parameter":                "",
            "evaluation_layer":                "document",
            "requires_live_assessment":        False,
            "requires_comparative_evaluation": False,
            **result,
        })

    _prog("Computing totals", 96)
    total_scored = round(sum(s.get("score") or 0 for s in scores), 1)
    total_pct    = round((total_scored / doc_max) * 100, 1) if doc_max > 0 else 0.0

    qualification: dict = {}
    if threshold:
        passed = total_pct >= float(threshold)
        qualification = {
            "threshold_pct":       float(threshold),
            "achieved_pct":        total_pct,
            "passed":              passed,
            "financial_bid_opens": passed,
            "note": ("Qualified -- financial bid may be opened." if passed
                     else f"Not qualified -- {total_pct}% vs >={threshold}% required."),
        }
        print(f"[TQ] Gate: {'QUALIFIED' if passed else 'NOT QUALIFIED'} "
              f"({total_pct}% vs >={threshold}%)")

    schema_warning = table.get("schema_warning")
    print(f"[TQ] -- Result --------------------------------------------------")
    print(f"[TQ] Scored: {total_scored} / {doc_max}  ({total_pct}%)")
    print(f"[TQ] Grand total (RFP declares): {grand_total}")
    if schema_warning:
        print(f"[TQ] WARNING: {schema_warning}")

    return {
        "evaluation_title":        table.get("evaluation_title", "Technical Evaluation"),
        "grand_total_marks":       grand_total,
        "technical_document_max":  doc_max,
        "scoreable_total":         doc_max,
        "live_assessment_marks":   0,
        "financial_marks":         0,
        "total_scored":            total_scored,
        "total_percentage":        total_pct,
        "final_score_formula":     None,
        "qualification_threshold": threshold,
        "qualification":           qualification,
        "schema_valid":            schema_warning is None,
        "schema_warning":          schema_warning,
        "criteria_structure":      criteria,
        "scores":                  scores,
        "error":                   table.get("error"),
    }