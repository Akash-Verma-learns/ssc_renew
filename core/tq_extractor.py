"""
TQ Extractor v14
================

Architecture (three clean stages)
-----------------------------------

STAGE 1 -- Page selection  (deterministic, zero LLM)
  Reuses the proven TOC + page-scoring stack from the stable extractor.
  Finds the exact page cluster that contains the evaluation table
  (e.g. pages 43-47).  Returns a list of page numbers.

STAGE 2 -- Table extraction  (Docling primary, LLM fallback)
  Extracts ONLY those pages from the PDF into a temporary in-memory PDF.
  Runs Docling's TableFormer on that mini-PDF.
  Docling returns the scoring table as a pandas DataFrame -- columns
  S.No | Parameter Name | Particulars/Criteria | Max. Marks | Document.
  A deterministic parser converts the DataFrame to criterion dicts.
  No LLM involved.  If Docling is not installed or fails, falls back
  to the LLM-based extraction used in v13 (proven reliable).

STAGE 3 -- Proposal scoring  (Python formula primary, tiny LLM for fact extraction)
  For each criterion:
    A. Detect formula type from criteria_text (STEP / BAND / PER_UNIT / QUAL / LLM)
    B. Ask a tiny LLM prompt (< 2 000 chars, 60 s timeout) for ONE specific fact.
       e.g. "What is the average annual turnover in Cr?"
    C. Apply the Python formula to that fact -- no LLM arithmetic.
  Scores are strictly grounded in proposal text.  If the fact is not found,
  score = 0 with a clear gap message.

Files to DELETE (no longer needed):
  core/tq_step1_extract.py   -- superseded by Stage 1 here
  core/tq_step2_score.py     -- superseded by Stage 3 here
  core/llm_client.py         -- SSC1 only; not used by TQ
  core/extractor_ssc.py      -- SSC1 experiment; not used by TQ
  core/parser_ssc.py         -- Docling parser is inline here
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import requests
import tempfile
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
OLLAMA_TIMEOUT   = 600          # long timeout for table-extraction fallback
OLLAMA_TIMEOUT_S = 60           # short timeout for single-fact extraction

TQ_UPLOAD_DIR = Path("./tq_uploads")
TQ_UPLOAD_DIR.mkdir(exist_ok=True)

_DB_PARAM_MAX        = 295
MAX_CONTEXT_CHARS    = 40_000
MAX_SECTION_SPAN     = 20
PAGE_SCORE_THRESHOLD = 5
HOT_THRESHOLD        = 7


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, ctx: int = 8192, timeout: int = OLLAMA_TIMEOUT) -> str:
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.0, "num_ctx": ctx},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        if not content:
            print(f"[TQ] Ollama returned empty (ctx={ctx})")
        return content or ""
    except requests.exceptions.Timeout:
        print(f"[TQ] Ollama timeout after {timeout}s")
        return ""
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


def _parse_json(text: str) -> Optional[dict]:
    cleaned = _clean_json(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


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
# Signal regexes (page scoring)
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
# STAGE 1 -- TOC parser + page scoring (deterministic)
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

    print("[TQ] TOC: eval section not found")
    return None, None


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


def _find_eval_cluster(rfp_doc_name: str) -> tuple[list, Optional[int], Optional[int]]:
    """
    Returns (cluster_pages, toc_start, toc_end).
    cluster_pages is the list of 0-based page INDICES for fitz.
    """
    all_chunks = _deduplicate(get_all_chunks_for_doc(rfp_doc_name))
    if not all_chunks:
        return [], None, None

    toc_start, toc_end = _parse_toc(all_chunks)
    rag                = _rag_pages(rfp_doc_name)
    print(f"[TQ] RAG hit pages: {sorted(rag)}")

    scores  = _score_pages(all_chunks, toc_start, toc_end, rag)
    top     = sorted([(pg, sc, rs) for pg, (sc, rs) in scores.items() if sc > 0],
                     key=lambda x: -x[1])[:15]
    print("[TQ] Top page scores:")
    for pg, sc, rs in top:
        print(f"     p{pg:3d}  score={sc:5.1f}  [{', '.join(rs)}]")

    cluster = _best_cluster(scores, toc_start)
    if cluster:
        print(f"[TQ] Cluster: {cluster}")
    else:
        print("[TQ] No cluster found")

    return cluster, toc_start, toc_end


# ---------------------------------------------------------------------------
# STAGE 2A -- Docling table extraction (primary)
# ---------------------------------------------------------------------------

def _extract_pages_as_pdf(pdf_path: str, page_numbers_1based: list) -> Optional[bytes]:
    """
    Extract specific pages from a PDF into a new in-memory PDF.
    page_numbers_1based: list of 1-based page numbers (as stored in chunks).
    Returns raw bytes of the new PDF, or None on failure.
    """
    try:
        import fitz
        src = fitz.open(pdf_path)
        dst = fitz.open()
        for pg_1 in page_numbers_1based:
            idx = pg_1 - 1   # fitz is 0-based
            if 0 <= idx < len(src):
                dst.insert_pdf(src, from_page=idx, to_page=idx)
        data = dst.tobytes()
        src.close(); dst.close()
        return data
    except Exception as e:
        print(f"[TQ] PDF page extraction failed: {e}")
        return None


def _docling_extract_tables(pdf_bytes: bytes) -> list:
    """
    Run Docling on a PDF (as bytes) and return list of TableInfo dicts:
      { page_no, dataframe, markdown }
    Returns [] if Docling is not installed or fails.
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        import pandas as pd
    except ImportError:
        print("[TQ] Docling not installed -- run: pip install docling")
        return []

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        opts.do_ocr             = False
        opts.table_structure_options.do_cell_matching = True
        opts.table_structure_options.mode = TableFormerMode.FAST

        conv   = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        result = conv.convert(tmp_path)
        doc    = result.document

        tables = []
        for tbl in doc.tables:
            page_no = (tbl.prov[0].page_no if getattr(tbl, "prov", None) else 0)
            try:
                df = tbl.export_to_dataframe(doc)
            except Exception:
                df = pd.DataFrame()
            try:
                md = tbl.export_to_markdown(doc)
            except Exception:
                md = df.to_string() if not df.empty else ""

            if not df.empty and len(df) >= 2:
                tables.append({"page_no": page_no, "dataframe": df, "markdown": md})

        Path(tmp_path).unlink(missing_ok=True)
        print(f"[TQ] Docling: found {len(tables)} tables in eval page cluster")
        return tables

    except Exception as e:
        print(f"[TQ] Docling extraction failed: {e}")
        try: Path(tmp_path).unlink(missing_ok=True)
        except Exception: pass
        return []


# ---------------------------------------------------------------------------
# STAGE 2B -- DataFrame -> criterion dicts (deterministic parser)
# ---------------------------------------------------------------------------

_SKIP_ROW_PATTERNS = re.compile(
    r"""(
        ^presentation$          |
        ^interview$             |
        viva\b                  |
        ^demo$                  |
        ^panel$                 |
        financial\s+bid         |
        price\s+bid             |
        \bL1\b                  |
        commercial\s+bid        |
        indemnity               |
        arbitration             |
        combined\s+and\s+final  |
        hiring\s+[&and]+\s+implementation |
        appreciation\s+and\s+response     |
        opening\s+of.*financial           |
        evaluation\s+of\s+financial
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_SUBITEM_PATTERNS = re.compile(
    r"^(team\s+leader|procurement\s+expert|documentation\s+expert|"
    r"urban\s+planning\s+expert|environmental\s+expert|animal\s+care\s+expert|"
    r"ict\s*/\s*it|gis\s+expert|data\s+analyst|legal\s+policy|"
    r"urban\s+finance|finance\s+expert|reporting\s+manager|liaison\s+officer|"
    r"ppp\s+specialist)",
    re.IGNORECASE,
)

_BAND_ROW_PATTERNS = re.compile(
    r"""(
        ^single\s+order\s+of\b      |
        ^for\s+every\s+additional\b |
        ^\d+\s+marks?\s+for\b       |
        ^order\s+copy\b             |
        ^audited\s+balance\b        |
        ^cvs?\s+of\s+the\b          |
        ^only\s+completion\b        |
        ^for\s+minimum\s+\d+\s+projects? |
        ^for\s+every\s+additional\s+project
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _find_column(df, keywords: list) -> Optional[int]:
    """Find the column index whose header best matches any keyword."""
    import pandas as pd
    for ki, kw in enumerate(keywords):
        for ci, col in enumerate(df.columns):
            if kw.lower() in str(col).lower():
                return ci
    # Also check first row as header
    if len(df) > 0:
        for ci, val in enumerate(df.iloc[0]):
            for kw in keywords:
                if kw.lower() in str(val).lower():
                    return ci
    return None


def _extract_max_marks_from_cell(cell_text: str) -> Optional[int]:
    """
    Extract the integer from the Max. Marks cell.
    Handles OCR artefacts: "15 mark s", "15\nmarks", plain "15".
    Strictly 1-60 range.
    """
    text = str(cell_text).strip()
    # Remove "marks" and variants to get the bare number
    cleaned = re.sub(r"mark\s*s?\b", "", text, flags=re.IGNORECASE).strip()
    # Find all integers in 1-60 range
    candidates = [int(m) for m in re.findall(r"\b(\d{1,2})\b", cleaned)
                  if 1 <= int(m) <= 60]
    if candidates:
        # Return the largest (most likely the actual max marks, not a sub-band)
        return max(candidates)
    return None


def _is_sno_cell(cell_text: str) -> bool:
    """True if this cell looks like a row number (1-20)."""
    t = str(cell_text).strip().rstrip(".")
    return bool(re.match(r"^\d{1,2}$", t)) and 1 <= int(t) <= 20


def _parse_dataframe_to_criteria(df, table_markdown: str = "") -> list:
    """
    Parse a Docling-extracted DataFrame into criterion dicts.
    Handles multi-row cells (Docling merges them) and OCR artefacts.
    Returns list of {item_code, parameter, max_marks, criteria_text}.
    """
    import pandas as pd

    # Normalise: stringify everything
    df = df.fillna("").astype(str)

    rows   = df.values.tolist()
    n_cols = df.shape[1]

    if n_cols < 3:
        print(f"[TQ] DataFrame too narrow ({n_cols} cols) -- skipping")
        return []

    # Heuristic column identification
    # Try to find S.No, Parameter, Criteria, Max.Marks columns by scanning
    # the first few rows for header-like content

    sno_col    = None
    param_col  = None
    crit_col   = None
    marks_col  = None

    # Scan first 4 rows as potential headers
    for row in rows[:4]:
        for ci, val in enumerate(row):
            v = str(val).lower().strip()
            if re.match(r"s\.?\s*no\.?|serial", v) and sno_col is None:
                sno_col = ci
            if re.match(r"param(eter)?(\s+name)?|criterion", v) and param_col is None:
                param_col = ci
            if re.match(r"particular|criteria|description", v) and crit_col is None:
                crit_col = ci
            if re.match(r"max|marks?|full\s+marks?", v) and marks_col is None:
                marks_col = ci

    # If column detection failed, apply positional heuristic for 5-col table
    if sno_col is None:
        sno_col = 0
    if param_col is None:
        param_col = 1
    if crit_col is None:
        crit_col = 2 if n_cols >= 4 else 1
    if marks_col is None:
        # Rightmost numeric-looking column
        marks_col = n_cols - 2 if n_cols >= 4 else n_cols - 1

    print(f"[TQ] DataFrame cols: sno={sno_col} param={param_col} "
          f"crit={crit_col} marks={marks_col}  (total={n_cols})")

    # Merge multi-row cells: group rows by the S.No anchor
    # A row belongs to criterion N if its sno cell is N or is blank
    criteria_raw: list[dict] = []
    current: Optional[dict]  = None

    for row in rows:
        sno_val    = str(row[sno_col]).strip().rstrip(".")
        param_val  = str(row[param_col]).strip()
        crit_val   = str(row[crit_col]).strip() if crit_col < len(row) else ""
        marks_val  = str(row[marks_col]).strip() if marks_col < len(row) else ""

        if _is_sno_cell(sno_val):
            # Start new criterion
            if current:
                criteria_raw.append(current)
            current = {
                "item_code":     sno_val,
                "parameter":     param_val,
                "criteria_text": crit_val,
                "marks_raw":     marks_val,
            }
        elif current is not None:
            # Continuation row -- append to current criterion
            if param_val and param_val not in current["parameter"]:
                current["parameter"]     += (" " + param_val).rstrip()
            if crit_val:
                current["criteria_text"] += (" " + crit_val)
            if marks_val and not current["marks_raw"]:
                current["marks_raw"]      = marks_val
            elif marks_val and current["marks_raw"] != marks_val:
                # Keep the first non-empty value that looks like a max mark
                if not re.search(r"\b\d{1,2}\b", current["marks_raw"]):
                    current["marks_raw"] = marks_val

    if current:
        criteria_raw.append(current)

    # Convert to validated criterion dicts
    criteria = []
    for c in criteria_raw:
        name  = re.sub(r"\s+", " ", c.get("parameter", "")).strip()
        ctext = re.sub(r"\s+", " ", c.get("criteria_text", "")).strip()
        marks = _extract_max_marks_from_cell(c.get("marks_raw", ""))
        code  = c.get("item_code", "")

        if not name or marks is None or marks < 1:
            continue
        if marks > 60:
            print(f"[TQ] Docling: dropped (marks={marks} > 60): {name[:60]}")
            continue
        if _SKIP_ROW_PATTERNS.search(name):
            print(f"[TQ] Docling: dropped (skip): {name[:60]}")
            continue
        if _SUBITEM_PATTERNS.match(name):
            print(f"[TQ] Docling: dropped (sub-item): {name[:60]}")
            continue
        if _BAND_ROW_PATTERNS.search(name):
            print(f"[TQ] Docling: dropped (band row): {name[:60]}")
            continue

        criteria.append({
            "item_code":     code,
            "parameter":     name,
            "max_marks":     marks,
            "criteria_text": ctext,
        })

    return criteria


def _select_best_table(tables: list) -> list:
    """
    From multiple Docling tables, pick the one most likely to be the scoring table.
    Heuristic: table that has the most rows with S.No-looking cells AND marks-looking cells.
    """
    best_score = -1
    best_crit  = []

    for tbl in tables:
        df   = tbl.get("dataframe")
        crit = _parse_dataframe_to_criteria(df, tbl.get("markdown", ""))
        doc_max = sum(c["max_marks"] for c in crit)
        # Score: number of valid criteria weighted by doc_max proximity to 100
        tbl_score = len(crit) * 10 + min(doc_max, 100)
        print(f"[TQ] Docling table: {len(crit)} valid criteria, doc_max={doc_max}, "
              f"tbl_score={tbl_score}")
        if tbl_score > best_score:
            best_score = tbl_score
            best_crit  = crit

    return best_crit


# ---------------------------------------------------------------------------
# STAGE 2C -- LLM fallback (only if Docling fails or yields nothing)
# ---------------------------------------------------------------------------

_TABLE_PROMPT = """\
You are reading pages from an Indian government RFP.
Extract ONLY the Technical Bid Evaluation scoring table.

TABLE STRUCTURE (columns left to right):
  S.No | Parameter Name | Particulars / Criteria | Max. Marks | Document required

CRITICAL RULES:
1. ONE row per S.No only. Count S.No values (1,2,3,...) and extract EXACTLY that many rows.
2. Max. Marks is from the rightmost integer column ONLY -- not from band text inside Particulars.
   Example: "Single order of 06 professionals: 10 marks | 07-12: 15 marks | >12: 20 marks | MAX 20"
   Correct: max_marks=20 (from Max. Marks column). Wrong: create 3 rows with 10/15/20.
3. Parameter Name is SHORT (the leftmost label column). Do NOT put Particulars text in parameter.
4. Qualifications row = ONE row with the expert sub-table inside criteria_text.
5. Skip: Presentation / Financial Bid / Contract clauses.

RFP PAGES:
{context}

Return ONLY valid JSON:
{{
  "evaluation_title": "section title",
  "grand_total_marks": <integer>,
  "qualification_threshold_pct": <number or null>,
  "visible_sno_values": [1,2,3,4,5],
  "skipped_rows": ["name -- reason"],
  "criteria": [
    {{
      "item_code": "<S.No>",
      "parameter": "<SHORT Parameter Name>",
      "max_marks": <integer from Max. Marks column>,
      "criteria_text": "<VERBATIM full Particulars/Criteria>"
    }}
  ]
}}
"""


def _llm_extract_from_context(context: str) -> list:
    """LLM fallback: extract criteria from raw text context."""
    prompt = _TABLE_PROMPT.format(context=_esc(context))
    print(f"[TQ] LLM fallback: sending {len(prompt)} chars (ctx=32768)")

    raw = _call_ollama(prompt, ctx=32768, timeout=OLLAMA_TIMEOUT)
    if not raw.strip():
        return []

    data = _parse_json(raw)
    if not data:
        return []

    visible_sno = data.get("visible_sno_values", [])
    raw_crit    = data.get("criteria", [])
    valid       = []

    for c in raw_crit:
        name  = (c.get("parameter") or "").strip()
        marks = int(c.get("max_marks") or 0)
        if not name or marks < 1 or marks > 60:
            continue
        if _SKIP_ROW_PATTERNS.search(name):
            print(f"[TQ] LLM-fallback: dropped (skip): {name[:60]}"); continue
        if _SUBITEM_PATTERNS.match(name):
            print(f"[TQ] LLM-fallback: dropped (sub-item): {name[:60]}"); continue
        if _BAND_ROW_PATTERNS.search(name):
            print(f"[TQ] LLM-fallback: dropped (band): {name[:60]}"); continue
        valid.append({
            "item_code":     c.get("item_code", ""),
            "parameter":     name,
            "max_marks":     marks,
            "criteria_text": (c.get("criteria_text") or "").strip(),
        })

    if visible_sno and len(valid) > len(visible_sno):
        print(f"[TQ] LLM-fallback WARNING: {len(valid)} criteria > "
              f"{len(visible_sno)} visible rows -- possible hallucination")

    return valid


def _build_text_context(all_chunks: list, cluster: list) -> str:
    """Build raw text context from cluster pages (for LLM fallback)."""
    page_set = set(cluster)
    chunks   = [c for c in all_chunks if c.get("page_no", 0) in page_set]
    parts, total = [], 0
    for c in sorted(chunks, key=lambda x: x.get("page_no", 0)):
        block = (f"[Page {c.get('page_no', 0)} | {c.get('section_heading', '')}]\n"
                 f"{c.get('text', '')}")
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block); total += len(block)
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main table extraction entry point
# ---------------------------------------------------------------------------

def _dedup_criteria(criteria: list) -> list:
    """Keep longest criteria_text for each (normalised_name, marks) pair."""
    seen: dict = {}
    for c in criteria:
        key = (re.sub(r"\s+", " ", c.get("parameter", "")).strip().lower(),
               int(c.get("max_marks") or 0))
        if key not in seen or len(c.get("criteria_text", "")) > len(seen[key].get("criteria_text", "")):
            seen[key] = c
    return list(seen.values())


def extract_marking_table(rfp_doc_name: str) -> dict:
    """
    Full Stage 1 + Stage 2 pipeline.
    Returns the standard table dict used by the orchestrator.
    """
    print(f"[TQ] Extracting marking table from: {rfp_doc_name}")

    # Stage 1: find eval page cluster
    all_chunks   = _deduplicate(get_all_chunks_for_doc(rfp_doc_name))
    cluster, toc_start, toc_end = _find_eval_cluster(rfp_doc_name)

    criteria: list = []
    source         = "unknown"

    # Stage 2A: Docling (preferred)
    pdf_path = Path("./uploads") / rfp_doc_name
    if cluster and pdf_path.exists():
        pdf_bytes = _extract_pages_as_pdf(str(pdf_path), cluster)
        if pdf_bytes:
            tables   = _docling_extract_tables(pdf_bytes)
            criteria = _select_best_table(tables)
            if criteria:
                source = f"Docling (cluster p{cluster[0]}-p{cluster[-1]})"
                print(f"[TQ] Docling: extracted {len(criteria)} criteria")

    # Stage 2B: LLM fallback
    if not criteria:
        print("[TQ] Falling back to LLM extraction")
        effective_cluster = cluster or list(range(
            max(1, (toc_start or 40)),
            (toc_end or (toc_start or 40) + 6) + 1
        ))
        context  = _build_text_context(all_chunks, effective_cluster)
        criteria = _llm_extract_from_context(context)
        source   = f"LLM-fallback (cluster p{effective_cluster[0] if effective_cluster else '?'})"

    if not criteria:
        return {"criteria": [], "grand_total_marks": 0,
                "error": "No criteria extracted.", "context_source": source}

    criteria    = _dedup_criteria(criteria)
    doc_max     = sum(c["max_marks"] for c in criteria)
    grand_total = doc_max   # will be overridden if LLM returned it

    schema_warning = None
    if doc_max < 20:
        schema_warning = f"Only {doc_max} marks extracted -- likely missed rows."

    print(f"[TQ] Final: {len(criteria)} criteria | doc_max={doc_max}")
    for c in criteria:
        print(f"  [{str(c.get('item_code','?')):3s}] "
              f"{c['parameter'][:55]:55s}  {c['max_marks']:3d} marks")

    return {
        "evaluation_title":            "Technical Evaluation",
        "grand_total_marks":           grand_total,
        "qualification_threshold_pct": 70.0,
        "criteria":                    criteria,
        "doc_max":                     doc_max,
        "schema_warning":              schema_warning,
        "context_source":              source,
        "error":                       None,
    }


# ---------------------------------------------------------------------------
# STAGE 3 -- Strict proposal scoring
# ---------------------------------------------------------------------------

# -- Formula detection -------------------------------------------------------

def _detect_formula(parameter: str, criteria_text: str) -> str:
    ct = criteria_text.lower()
    p  = parameter.lower()

    if re.search(r"every\s+additional\s+\d+\s*cr", ct, re.I):
        return "STEP"
    if len(re.findall(r"\d+\s*professional", ct, re.I)) >= 2:
        return "BAND"
    if re.search(r"\d+\s*marks?\s+for\s+(?:\d+|each|01|per)\s+(?:project|assignment)", ct, re.I):
        return "PER_UNIT"
    if re.search(r"(\d+%.*?(?:education|experience|project|qualification))", ct, re.I):
        return "QUAL"
    if "qualification" in p and ("competence" in p or "staff" in p):
        return "QUAL"
    return "LLM"


# -- Proposal page retrieval (keyword-scored, no ChromaDB) --------------------

_KW_SETS = {
    "turnover":      ["turnover", "crore", "annual", "revenue", "balance sheet",
                      "financial year", "average annual"],
    "professionals": ["professional", "manpower", "order", "employees", "advisory",
                      "consulting", "supply", "deployed", "staffing"],
    "projects":      ["project", "assignment", "pmc", "pmu", "urban", "amrut",
                      "smart city", "completion certificate", "ulb", "billing"],
    "qualification": ["cv", "curriculum vitae", "qualification", "years of experience",
                      "team leader", "expert", "education", "degree", "relevant experience"],
    "methodology":   ["methodology", "approach", "work plan", "implementation", "strategy"],
    "generic":       ["experience", "project", "relevant", "work", "assignment"],
}


def _pick_kw_set(parameter: str, criteria_text: str) -> list:
    combined = (parameter + " " + criteria_text).lower()
    if any(w in combined for w in ["turnover", "crore", "annual"]):
        return _KW_SETS["turnover"]
    if any(w in combined for w in ["professional", "manpower", "employees in supply"]):
        return _KW_SETS["professionals"]
    if any(w in combined for w in ["pmc", "pmu", "urban", "amrut", "billing", "consulting"]):
        return _KW_SETS["projects"]
    if any(w in combined for w in ["qualification", "competence", "cv", "staff"]):
        return _KW_SETS["qualification"]
    if any(w in combined for w in ["methodology", "approach", "work plan"]):
        return _KW_SETS["methodology"]
    return _KW_SETS["generic"]


def _get_proposal_pages(proposal_path: str, parameter: str,
                        criteria_text: str, max_chars: int = 2_500) -> str:
    """Open proposal PDF directly with fitz; score pages by keyword hits."""
    try:
        import fitz
    except ImportError:
        return ""

    kws    = _pick_kw_set(parameter, criteria_text)
    doc    = fitz.open(proposal_path)
    scored = []
    for pno in range(len(doc)):
        txt  = doc[pno].get_text()
        low  = txt.lower()
        hits = sum(1 for kw in kws if kw in low)
        if hits > 0:
            scored.append((hits, pno + 1, txt.strip()))
    scored.sort(reverse=True)
    doc.close()

    if not scored:
        # Fallback: first 6 pages
        doc2   = fitz.open(proposal_path)
        parts  = [f"[Page {i+1}]\n{doc2[i].get_text().strip()[:500]}"
                  for i in range(min(6, len(doc2)))]
        doc2.close()
        return "\n\n".join(parts)[:max_chars]

    parts, total = [], 0
    for _, pno, txt in scored[:3]:
        block = f"[Page {pno}]\n{txt}"
        if total + len(block) > max_chars:
            block = block[:max_chars - total]
        parts.append(block); total += len(block)
        if total >= max_chars: break

    return "\n\n---\n\n".join(parts)


# -- Fact extraction ----------------------------------------------------------

_EXTRACT_PROMPT = """\
Read the proposal pages and find ONE specific piece of information.

FIND: {what}

PROPOSAL PAGES:
{pages}

Return ONLY valid JSON:
{{"found": true or false, "value": "<exact value e.g. 180 Cr or 8 professionals or 3 projects>", "page": <integer or null>}}

If not found: {{"found": false, "value": null, "page": null}}"""


def _what_to_find(parameter: str, criteria_text: str) -> str:
    combined = (parameter + " " + criteria_text).lower()
    if "turnover" in combined:
        return ("the bidder's average annual turnover in Indian Rupees Crore (INR Cr) "
                "over the last 3 financial years -- single number in Crore")
    if any(w in combined for w in ["professional", "manpower", "employees in supply"]):
        return ("the maximum number of professionals / employees supplied in a SINGLE "
                "work order for advisory or consulting services -- single integer count")
    if any(w in combined for w in ["pmc", "pmu", "urban", "amrut", "billing", "consulting",
                                    "assignment", "project"]):
        return ("the number of eligible consulting / PMC / PMU assignments "
                "with client billing of at least Rs 0.4 Cr per assignment -- single integer count")
    if any(w in combined for w in ["qualification", "competence", "cv", "staff"]):
        return ("for each proposed team member: name, role, highest educational qualification, "
                "years of relevant experience -- list all")
    if any(w in combined for w in ["methodology", "approach", "work plan"]):
        return ("whether the proposal contains a technical methodology or work plan -- "
                "YES with page reference, or NO")
    return f"the specific information required by the criterion: {parameter}"


def _extract_fact(proposal_path: str, parameter: str, criteria_text: str) -> dict:
    """Returns {found, value, page}."""
    pages  = _get_proposal_pages(proposal_path, parameter, criteria_text)
    if not pages:
        return {"found": False, "value": None, "page": None}

    what   = _what_to_find(parameter, criteria_text)
    prompt = _EXTRACT_PROMPT.format(what=what, pages=pages)
    print(f"    [fact] prompt={len(prompt)} chars")

    raw    = _call_ollama(prompt, ctx=4096, timeout=OLLAMA_TIMEOUT_S)
    result = _parse_json(raw) if raw else None
    if result:
        return {
            "found": bool(result.get("found")),
            "value": str(result["value"]) if result.get("value") else None,
            "page":  result.get("page"),
        }
    return {"found": False, "value": None, "page": None}


# -- Python formula implementations ------------------------------------------

def _step_formula(criteria_text: str, max_marks: int, value_str: str) -> Optional[float]:
    """Turnover step scoring: base + increments per extra N Cr."""
    base_m = re.search(
        r"(\d+(?:\.\d+)?)\s*cr[ores]*\s*[.=:\-\s]+\s*(\d+(?:\.\d+)?)\s*marks?",
        criteria_text, re.IGNORECASE,
    )
    step_m = re.search(
        r"(?:every|each|per)\s+additional\s+(\d+(?:\.\d+)?)\s*cr[ores]*"
        r"[\s\W]*(\d+(?:\.\d+)?)\s*marks?",
        criteria_text, re.IGNORECASE,
    )
    if not (base_m and step_m):
        return None
    try:
        base_threshold = float(base_m.group(1))
        base_score     = float(base_m.group(2))
        step_size      = float(step_m.group(1))
        step_score     = float(step_m.group(2))
        nums = re.findall(r"[\d,]+(?:\.\d+)?", value_str.replace(",", ""))
        if not nums: return None
        turnover = float(nums[0])
        if turnover < base_threshold: return 0.0
        extra_steps = int((turnover - base_threshold) / step_size)
        return round(min(base_score + extra_steps * step_score, max_marks), 1)
    except (ValueError, ZeroDivisionError):
        return None


def _band_formula(criteria_text: str, max_marks: int, value_str: str) -> Optional[float]:
    """Professionals band scoring: ordered thresholds."""
    ct = criteria_text.lower()
    # Build bands from "N professionals : M marks" and "more than N professionals : M marks"
    simple_bands = re.findall(
        r"(?:of\s+)?(\d+)\s+professionals?\s*[:\-]\s*(\d+)\s*marks?", ct, re.IGNORECASE
    )
    range_bands = re.findall(
        r"more\s+than\s+(\d+)(?:[\s\-]*and[\s\-]*up[\s\-]*to\s+(\d+))?\s+"
        r"professionals?\s*[:\-]\s*(\d+)\s*marks?", ct, re.IGNORECASE
    )
    bands = []
    for cnt, mk in simple_bands:
        bands.append((int(cnt), int(mk)))
    for lo, hi, mk in range_bands:
        upper = int(hi) if hi else 9999
        bands.append((upper, int(mk)))
    if not bands: return None
    bands.sort(key=lambda b: b[0])
    try:
        nums = re.findall(r"\d+", value_str)
        if not nums: return None
        count = int(nums[0])
        score = float(bands[-1][1])
        for upper, mk in bands:
            if count <= upper:
                score = float(mk); break
        return round(min(score, max_marks), 1)
    except (ValueError, IndexError):
        return None


def _per_unit_formula(criteria_text: str, max_marks: int, value_str: str) -> Optional[float]:
    """Per-project scoring: N marks per project, capped."""
    rate_m = re.search(
        r"(\d+(?:\.\d+)?)\s*marks?\s+for\s+(?:\d+|each|01|per)\s+(?:project|assignment)",
        criteria_text, re.IGNORECASE,
    )
    if not rate_m: return None
    try:
        rate = float(rate_m.group(1))
        nums = re.findall(r"\d+", value_str)
        if not nums: return None
        count = int(nums[0])
        return round(min(count * rate, max_marks), 1)
    except (ValueError, ZeroDivisionError):
        return None


def _qual_formula(proposal_path: str, parameter: str,
                  criteria_text: str, max_marks: int) -> tuple[float, str]:
    """Qualifications: structured evidence check with a small LLM call."""
    pages  = _get_proposal_pages(proposal_path, parameter, criteria_text, max_chars=1_500)
    prompt = f"""Check if the proposal includes the following. Answer YES or NO for each.

1. Named team members or proposed experts?
2. Educational qualifications stated (degree, diploma, certification)?
3. Years of relevant experience stated for each expert?
4. Relevant projects listed for the proposed team?
5. CVs or detailed profiles attached?

PROPOSAL:
{pages}

Return ONLY valid JSON:
{{"q1": true/false, "q2": true/false, "q3": true/false, "q4": true/false, "q5": true/false, "note": "one sentence"}}"""

    raw    = _call_ollama(prompt, ctx=4096, timeout=OLLAMA_TIMEOUT_S)
    result = _parse_json(raw) if raw else {}
    if not result:
        return 0.0, "Qualification evidence check failed"

    weights = {"q1": 0.10, "q2": 0.20, "q3": 0.20, "q4": 0.30, "q5": 0.20}
    total_w = sum(w for k, w in weights.items() if result.get(k, False))
    score   = round(total_w * max_marks, 1)
    return score, result.get("note", "")


def _llm_score_formula(proposal_path: str, parameter: str, criteria_text: str,
                        max_marks: int, extracted: dict) -> tuple[float, str]:
    """LLM scoring for methodology / open-ended criteria."""
    pages = _get_proposal_pages(proposal_path, parameter, criteria_text, max_chars=1_200)
    ev    = extracted.get("value") or "Not explicitly found"
    pg    = extracted.get("page") or "unknown"
    rule  = criteria_text[:300]

    prompt = f"""Score a vendor proposal against one RFP criterion.

CRITERION: {parameter}
MAX MARKS: {max_marks}
SCORING RULE: {rule}

WHAT WAS FOUND: {ev} (page {pg})

PROPOSAL EXCERPT:
{pages[:800]}

Score 0 to {max_marks}. Base on quality and relevance of evidence found.

Return ONLY valid JSON:
{{"score": <0 to {max_marks}>, "justification": "one sentence"}}"""

    raw    = _call_ollama(prompt, ctx=4096, timeout=OLLAMA_TIMEOUT_S)
    result = _parse_json(raw) if raw else {}
    if not result:
        return 0.0, "LLM scoring failed"
    try:
        score = round(max(0.0, min(float(result.get("score") or 0), float(max_marks))), 1)
        return score, result.get("justification", "")
    except (TypeError, ValueError):
        return 0.0, "Score conversion error"


# -- Main criterion scorer ----------------------------------------------------

def score_criterion(criterion: dict, proposal_path: str) -> dict:
    """
    Score one criterion against the proposal.
    Returns the standard result dict.
    """
    max_marks     = int(criterion.get("max_marks") or 0)
    parameter     = criterion.get("parameter", "")
    criteria_text = criterion.get("criteria_text", "")

    def _zero(reason: str) -> dict:
        return {"score": 0, "extracted_value": None, "source_page": None,
                "scoring_steps": reason, "justification": reason,
                "strengths": [], "gaps": [reason], "evidence_found": False}

    if max_marks == 0:
        return _zero("Zero-mark criterion")

    if not Path(proposal_path).exists():
        return _zero(f"Proposal file not found: {proposal_path}")

    formula = _detect_formula(parameter, criteria_text)
    print(f"    [formula] {formula}")

    # Qualifications: structured check, no single-fact extraction needed
    if formula == "QUAL":
        score, note = _qual_formula(proposal_path, parameter, criteria_text, max_marks)
        return {
            "score":           score,
            "extracted_value": note or "Qualifications evidence check",
            "source_page":     None,
            "scoring_steps":   f"Structured evidence check -> {score}/{max_marks}",
            "justification":   note or f"Score {score}/{max_marks} based on CV/qualifications evidence",
            "strengths":       [note] if note and score > 0 else [],
            "gaps":            [] if score >= max_marks * 0.8 else ["Full marks require detailed CVs for all roles"],
            "evidence_found":  score > 0,
        }

    # Stage A: extract the specific fact
    extracted = _extract_fact(proposal_path, parameter, criteria_text)
    ev        = extracted.get("value") or "Not found"
    pg        = extracted.get("page")
    found     = extracted.get("found", False)
    print(f"    [fact] found={found}  value={ev!r}  page={pg}")

    if not found:
        if formula == "LLM":
            score, just = _llm_score_formula(
                proposal_path, parameter, criteria_text, max_marks, extracted)
            return _make_result(score, ev, pg, f"LLM: {just}", max_marks)
        return _zero(f"Key fact not found: {_what_to_find(parameter, criteria_text)[:80]}")

    # Stage B: apply Python formula
    python_score: Optional[float] = None
    steps = ""

    if formula == "STEP":
        python_score = _step_formula(criteria_text, max_marks, ev)
        steps = f"STEP formula: value={ev} -> {python_score}/{max_marks}"
    elif formula == "BAND":
        python_score = _band_formula(criteria_text, max_marks, ev)
        steps = f"BAND formula: value={ev} -> {python_score}/{max_marks}"
    elif formula == "PER_UNIT":
        python_score = _per_unit_formula(criteria_text, max_marks, ev)
        steps = f"PER_UNIT formula: value={ev} -> {python_score}/{max_marks}"
    else:
        score, just = _llm_score_formula(
            proposal_path, parameter, criteria_text, max_marks, extracted)
        return _make_result(score, ev, pg, f"LLM: {just}", max_marks)

    if python_score is not None:
        print(f"    [Python] {steps}")
        return _make_result(python_score, ev, pg, steps, max_marks)

    # Python formula parse failed -- LLM fallback
    print(f"    [LLM fallback] formula parse failed for {formula}")
    score, just = _llm_score_formula(
        proposal_path, parameter, criteria_text, max_marks, extracted)
    return _make_result(score, ev, pg, f"LLM fallback: {just}", max_marks)


def _make_result(score: float, ev: str, pg: Optional[int],
                 steps: str, max_marks: int) -> dict:
    score = round(max(0.0, min(score, float(max_marks))), 1)
    return {
        "score":           score,
        "extracted_value": ev,
        "source_page":     pg,
        "scoring_steps":   steps,
        "justification":   f"Score {score}/{max_marks}. Found: {ev}" + (f" (p.{pg})" if pg else ""),
        "strengths":       [f"Found: {ev}" + (f" (p.{pg})" if pg else "")] if score > 0 else [],
        "gaps":            [] if score >= max_marks else ["Additional evidence needed for full marks"],
        "evidence_found":  score > 0,
    }


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

    doc_max   = table.get("doc_max", sum(c["max_marks"] for c in criteria))
    threshold = table.get("qualification_threshold_pct", 70.0)

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
            result = score_criterion(criterion, proposal_path)
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
    if schema_warning:
        print(f"[TQ] WARNING: {schema_warning}")

    return {
        "evaluation_title":        table.get("evaluation_title", "Technical Evaluation"),
        "grand_total_marks":       table.get("grand_total_marks", doc_max),
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