"""
table_pipeline.py
LLM-summarized, dual-representation table chunking pipeline.

For each table:
  1. Parse markdown → detect headers, row/col count, column types
  2. LLM summary via Ollama qwen3 (or any configured text model)
  3. Dual-representation chunking:
     - 1 full_table chunk  (full markdown + summary → for comparison / aggregate queries)
     - N row_group chunks  (10 rows, 2-row overlap  → for specific value lookup)

All LLM calls are optional: pass llm_client=None to skip summaries gracefully.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TableChunk:
    chunk_id: str
    text: str           # markdown table content (full or row-group)
    chunk_type: str     # "full_table" | "row_group"
    llm_summary: str    # always present; "" if LLM was skipped / failed
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    return "tbl_" + uuid.uuid4().hex[:8]


def _recency_score(upload_date: str | None) -> float:
    if not upload_date:
        return 1.0
    try:
        if isinstance(upload_date, (date, datetime)):
            d = upload_date
        else:
            d = datetime.fromisoformat(str(upload_date)).date()
        days = (date.today() - (d if isinstance(d, date) else d.date())).days
        return round(math.exp(-days / 180), 4)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Markdown table parser
# ---------------------------------------------------------------------------

def _parse_markdown_table(md: str) -> dict:
    """
    Parse a GitHub-flavored markdown table string.

    Returns:
        {
          "header_names": list[str],
          "has_header": bool,
          "rows": list[list[str]],   # data rows only (no header row)
          "row_count": int,
          "col_count": int,
          "has_numbers": bool,
          "column_types": list[str], # "numeric" | "text" for each column
        }
    """
    lines = [l.rstrip() for l in md.strip().splitlines() if l.strip()]
    if not lines:
        return {"header_names": [], "has_header": False, "rows": [],
                "row_count": 0, "col_count": 0, "has_numbers": False, "column_types": []}

    def _split_row(line: str) -> list[str]:
        line = line.strip().strip("|")
        return [cell.strip() for cell in line.split("|")]

    # Detect separator row (e.g. |---|---|)
    sep_pattern = re.compile(r'^[\|\s\-:]+$')
    sep_idx = None
    for i, line in enumerate(lines):
        if sep_pattern.match(line) and i > 0:
            sep_idx = i
            break

    if sep_idx is not None:
        header_row = _split_row(lines[0])
        data_rows = [_split_row(l) for l in lines[sep_idx + 1:] if l.strip()]
        has_header = True
    else:
        header_row = []
        data_rows = [_split_row(l) for l in lines]
        has_header = False

    col_count = max((len(r) for r in ([header_row] if header_row else []) + data_rows), default=0)

    # Determine column types
    col_types: list[str] = []
    numeric_re = re.compile(r'^[\d,.$%\-+()]+$')
    for col_i in range(col_count):
        vals = [r[col_i] for r in data_rows if col_i < len(r) and r[col_i]]
        numeric_count = sum(1 for v in vals if numeric_re.match(v.replace(" ", "")))
        col_types.append("numeric" if vals and numeric_count / len(vals) >= 0.6 else "text")

    has_numbers = any(t == "numeric" for t in col_types) or bool(
        re.search(r'\d', md)
    )

    return {
        "header_names": header_row,
        "has_header": has_header,
        "rows": data_rows,
        "row_count": len(data_rows),
        "col_count": col_count,
        "has_numbers": has_numbers,
        "column_types": col_types,
    }


def _rebuild_markdown(header_names: list[str], rows: list[list[str]]) -> str:
    """Reconstruct a markdown table from header + selected rows."""
    if not rows:
        return ""

    col_count = max(len(header_names), max(len(r) for r in rows))

    def _pad(row: list[str], n: int) -> list[str]:
        return row + [""] * (n - len(row))

    lines: list[str] = []
    if header_names:
        lines.append("| " + " | ".join(_pad(header_names, col_count)) + " |")
        lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_pad(row, col_count)) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM summary via Ollama
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """\
Summarize the following table in 2-3 sentences. Describe:
- What the table represents (its subject/topic)
- Key values, trends, or comparisons visible
- Any notable outliers or patterns

Be concise and factual. Do not invent data not present in the table.

TABLE:
{table_markdown}
"""

def _ollama_summarize(table_markdown: str, model: str = "qwen3") -> str:
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(
                table_markdown=table_markdown[:4000]  # cap input to avoid context overflow
            )}],
            options={"num_predict": 200, "temperature": 0.1},
        )
        # Handle both dict and object response formats
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "").strip()
        return response.message.content.strip()
    except Exception:
        return ""


def _get_summary(table_markdown: str, llm_client: Any) -> str:
    """Route to the right LLM backend. Returns "" on failure."""
    if llm_client is None:
        return ""

    if llm_client == "ollama" or (isinstance(llm_client, dict) and
                                   llm_client.get("backend") == "ollama"):
        model = llm_client.get("model", "qwen3") if isinstance(llm_client, dict) else "qwen3"
        return _ollama_summarize(table_markdown, model)

    # Callable interface: client(prompt) → str
    if callable(llm_client):
        try:
            return llm_client(_SUMMARY_PROMPT.format(table_markdown=table_markdown[:4000]))
        except Exception:
            return ""

    return ""


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def _base_meta(
    chunk_id: str,
    chunk_type: str,
    table_index: int,
    parsed: dict,
    llm_summary: str,
    source_type: str,
    doc_meta: dict,
    row_start: int | None = None,
    row_end: int | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_meta.get("doc_id", ""),
        "dept_id": doc_meta.get("dept_id", ""),
        "doc_type": source_type,
        "content_type": "table",
        "chunk_type": chunk_type,
        "chunk_strategy": "dual_representation",
        "table_index": table_index,
        "row_count": parsed["row_count"],
        "col_count": parsed["col_count"],
        "has_header": parsed["has_header"],
        "header_names": parsed["header_names"],
        "column_types": parsed["column_types"],
        "has_numbers": parsed["has_numbers"],
        "row_start": row_start,
        "row_end": row_end,
        "llm_summary": llm_summary,
        "filename": doc_meta.get("filename", ""),
        "upload_date": str(doc_meta.get("upload_date", "")),
        "created_by": doc_meta.get("created_by", ""),
        "recency_score": _recency_score(doc_meta.get("upload_date")),
    }


# ---------------------------------------------------------------------------
# Dual-representation chunking
# ---------------------------------------------------------------------------

ROW_WINDOW = 10
ROW_OVERLAP = 2


def _dual_chunk_table(
    md: str,
    table_index: int,
    source_type: str,
    doc_meta: dict,
    llm_summary: str,
) -> list[TableChunk]:
    parsed = _parse_markdown_table(md)
    chunks: list[TableChunk] = []

    # --- Chunk A: full table ---
    cid = _make_id()
    full_text = md.strip()
    if llm_summary:
        full_text = f"Summary: {llm_summary}\n\n{full_text}"

    chunks.append(TableChunk(
        chunk_id=cid,
        text=full_text,
        chunk_type="full_table",
        llm_summary=llm_summary,
        metadata=_base_meta(cid, "full_table", table_index, parsed,
                            llm_summary, source_type, doc_meta,
                            row_start=0, row_end=parsed["row_count"]),
    ))

    # --- Chunks B: row-group chunks (skip if table is small enough already) ---
    rows = parsed["rows"]
    if len(rows) > ROW_WINDOW:
        step = ROW_WINDOW - ROW_OVERLAP
        for start in range(0, len(rows), step):
            end = min(start + ROW_WINDOW, len(rows))
            window_rows = rows[start:end]
            window_md = _rebuild_markdown(parsed["header_names"], window_rows)
            if not window_md:
                continue

            cid = _make_id()
            chunks.append(TableChunk(
                chunk_id=cid,
                text=window_md,
                chunk_type="row_group",
                llm_summary=llm_summary,
                metadata=_base_meta(cid, "row_group", table_index, parsed,
                                    llm_summary, source_type, doc_meta,
                                    row_start=start, row_end=end),
            ))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_tables(
    tables: list[str],
    source_type: str,
    doc_meta: dict,
    llm_client: Any = None,
) -> list[TableChunk]:
    """
    Chunk and tag tables from a parsed document.

    Args:
        tables:      List of markdown table strings from ParsedDocument.tables.
        source_type: File source type — "pdf" | "docx" | "pptx" | "url" | etc.
        doc_meta:    Document metadata dict with keys:
                       doc_id, dept_id, filename, upload_date, created_by.
        llm_client:  LLM backend for table summarization. Options:
                       - None              → skip summaries
                       - "ollama"          → use Ollama with default model (qwen3)
                       - {"backend": "ollama", "model": "qwen3"} → Ollama with custom model
                       - callable(prompt→str) → custom LLM function

    Returns:
        List of TableChunk objects (1 full_table + N row_group per table).
    """
    all_chunks: list[TableChunk] = []

    for idx, md in enumerate(tables):
        md = md.strip()
        if not md:
            continue

        # Summarize first so the summary is embedded in full_table chunk
        summary = _get_summary(md, llm_client)
        chunks = _dual_chunk_table(md, idx, source_type, doc_meta, summary)
        all_chunks.extend(chunks)

    return all_chunks
