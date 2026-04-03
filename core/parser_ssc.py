"""
core/parser.py
--------------
Document parser: Docling primary (table-aware), PyMuPDF fallback.

WHY DOCLING
-----------
The root cause of every TQ extraction failure is that PDF tables are split
across chunks by PyMuPDF. A scoring table spanning pages 43-46 becomes
30 separate text fragments that the LLM has to mentally re-assemble — and
frequently gets wrong.

Docling's TableFormer model detects tables at the layout level and returns
them as structured rows+columns (pandas DataFrames). The scoring table on
pages 43-46 comes back as ONE DataFrame with columns:
    S.No | Parameter Name | Particulars Criteria | Max. Marks | Document required

No chunk scoring, no page proximity math, no LLM table extraction needed.

OUTPUT
------
parse_document(path)              → list[Chunk]          (for ChromaDB / semantic search)
parse_document_with_tables(path)  → (list[Chunk], list[TableInfo])
                                    TableInfo = {page_no, dataframe, markdown, text}
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Chunk dataclass — same shape as before (ChromaDB compatible)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text:            str
    page_no:         int
    section_heading: str = ""
    clause_ref:      str = ""
    doc_name:        str = ""
    chunk_id:        str = field(default="")

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = _make_id(self.doc_name, self.page_no, self.text)


def _make_id(doc_name: str, page_no: int, text: str) -> str:
    h = hashlib.md5(text.strip().lower().encode()).hexdigest()[:8]
    return f"{doc_name}_{page_no}_{h}"


@dataclass
class TableInfo:
    page_no:   int
    dataframe: object          # pandas DataFrame
    markdown:  str             # plain-text representation for LLM fallback
    text:      str             # raw concatenated cell text


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_document(file_path: str, doc_name: str = "") -> list[Chunk]:
    """Return text chunks for ChromaDB ingestion."""
    chunks, _ = parse_document_with_tables(file_path, doc_name=doc_name)
    return chunks


def parse_document_with_tables(
    file_path: str,
    doc_name:  str = "",
) -> tuple[list[Chunk], list[TableInfo]]:
    """
    Parse a PDF/DOCX and return:
      - text chunks (for ChromaDB)
      - structured tables (for TQ scoring table extraction)

    Docling is tried first; PyMuPDF is the fallback.
    """
    doc_name = doc_name or Path(file_path).name

    try:
        return _parse_with_docling(file_path, doc_name)
    except ImportError:
        print(f"[Parser] Docling not installed — using PyMuPDF fallback. "
              f"Run: pip install docling")
    except Exception as e:
        print(f"[Parser] Docling failed ({e}) — using PyMuPDF fallback")

    chunks = _parse_with_pymupdf(file_path, doc_name)
    return chunks, []


# ─────────────────────────────────────────────────────────────────────────────
# Docling parser (primary)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_with_docling(
    file_path: str,
    doc_name:  str,
) -> tuple[list[Chunk], list[TableInfo]]:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr             = False
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST  # ADD THIS

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    result  = converter.convert(file_path)
    doc     = result.document

    chunks: list[Chunk]     = []
    tables: list[TableInfo] = []
    current_heading         = ""
    chunk_idx               = 0

    # ── Text items (paragraphs, headers, list items) ──────────────────────────
    for item, _ in doc.iterate_items():
        label    = getattr(item, "label", "")
        text     = getattr(item, "text", "").strip()
        page_no  = (item.prov[0].page_no if getattr(item, "prov", None) else 0)

        if not text:
            continue

        str_label = str(label).lower()

        if "section_header" in str_label or "title" in str_label:
            current_heading = text

        if "table" in str_label:
            # Tables are handled separately below
            continue

        if len(text) < 15:
            # Skip very short fragments (page numbers, stray chars, etc.)
            continue

        chunk_idx += 1
        chunks.append(Chunk(
            text            = text,
            page_no         = page_no,
            section_heading = current_heading,
            clause_ref      = f"p{page_no}_{chunk_idx}",
            doc_name        = doc_name,
            chunk_id        = f"{doc_name}_{page_no}_{chunk_idx:04d}",
        ))

    # ── Tables ────────────────────────────────────────────────────────────────
    import pandas as pd

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

        raw_text = md.replace("|", " ").replace("-", " ")
        raw_text = re.sub(r"\s+", " ", raw_text).strip()

        if df.empty or len(df) < 2:
            continue

        tables.append(TableInfo(
            page_no   = page_no,
            dataframe = df,
            markdown  = md,
            text      = raw_text,
        ))

        # Also add table as a text chunk so semantic search can find it
        chunk_idx += 1
        chunks.append(Chunk(
            text            = f"[TABLE page {page_no}]\n{raw_text}",
            page_no         = page_no,
            section_heading = current_heading,
            clause_ref      = f"p{page_no}_table",
            doc_name        = doc_name,
            chunk_id        = f"{doc_name}_{page_no}_table_{chunk_idx:04d}",
        ))

    print(f"[Parser/Docling] {len(chunks)} chunks, {len(tables)} tables "
          f"from '{Path(file_path).name}'")

    return chunks, tables


# ─────────────────────────────────────────────────────────────────────────────
# PyMuPDF fallback parser
# ─────────────────────────────────────────────────────────────────────────────

_HEADING_RE = re.compile(
    r"^\s*(\d+[\.\)]\s+|[A-Z]{2,}[\s:]+|clause\s+\d|section\s+\d)",
    re.IGNORECASE,
)

MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 1200


def _parse_with_pymupdf(file_path: str, doc_name: str) -> list[Chunk]:
    import fitz  # PyMuPDF

    doc     = fitz.open(file_path)
    chunks: list[Chunk] = []
    current_heading     = ""
    chunk_idx           = 0

    for page in doc:
        page_no    = page.number + 1
        blocks     = page.get_text("blocks", sort=True)
        page_buf   = []

        for (x0, y0, x1, y1, text, block_no, block_type) in blocks:
            if block_type != 0:
                continue
            text = text.strip()
            if not text or len(text) < 8:
                continue

            if _HEADING_RE.match(text) and len(text) < 120:
                # Flush buffer before heading
                if page_buf:
                    chunk_idx += 1
                    full = " ".join(page_buf)
                    chunks.append(Chunk(
                        text            = full[:MAX_CHUNK_CHARS],
                        page_no         = page_no,
                        section_heading = current_heading,
                        clause_ref      = f"p{page_no}_{chunk_idx}",
                        doc_name        = doc_name,
                        chunk_id        = f"{doc_name}_{page_no}_{chunk_idx:04d}",
                    ))
                    page_buf = []
                current_heading = text

            page_buf.append(text)

            # Flush when buffer is large enough
            if sum(len(t) for t in page_buf) >= MAX_CHUNK_CHARS:
                chunk_idx += 1
                full = " ".join(page_buf)
                chunks.append(Chunk(
                    text            = full[:MAX_CHUNK_CHARS],
                    page_no         = page_no,
                    section_heading = current_heading,
                    clause_ref      = f"p{page_no}_{chunk_idx}",
                    doc_name        = doc_name,
                    chunk_id        = f"{doc_name}_{page_no}_{chunk_idx:04d}",
                ))
                page_buf = []

        # Flush remaining
        if page_buf and sum(len(t) for t in page_buf) >= MIN_CHUNK_CHARS:
            chunk_idx += 1
            full = " ".join(page_buf)
            chunks.append(Chunk(
                text            = full[:MAX_CHUNK_CHARS],
                page_no         = page_no,
                section_heading = current_heading,
                clause_ref      = f"p{page_no}_{chunk_idx}",
                doc_name        = doc_name,
                chunk_id        = f"{doc_name}_{page_no}_{chunk_idx:04d}",
            ))

    doc.close()
    print(f"[Parser/PyMuPDF] {len(chunks)} chunks from '{Path(file_path).name}'")
    return chunks