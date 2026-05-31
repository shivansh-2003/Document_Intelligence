"""
table_pipeline.py
LLM-summarized, dual-representation table chunking pipeline.

Model: qwen3.6 (single model — replaces separate qwen3 text model)

For each table:
  1. Parse markdown → detect headers, row/col count, column types
     (with fallback for space-separated tables from unstructured)
  2. LLM summary via qwen3.6
  3. Dual-representation chunking:
     - 1 full_table chunk  → entire table + summary (comparison / aggregate queries)
     - N row_group chunks  → 10-row windows, 2-row overlap (specific value lookup)
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TableChunk:
    chunk_id:    str
    text:        str        # markdown table content (full or row-group)
    chunk_type:  str        # "full_table" | "row_group"
    llm_summary: str        # "" if LLM was skipped / failed
    metadata:    dict[str, Any] = field(default_factory=dict)


DEFAULT_MODEL = "qwen3.6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    return "tbl_" + uuid.uuid4().hex[:8]


def _recency_score(upload_date: str | None) -> float:
    if not upload_date:
        return 1.0
    try:
        d = datetime.fromisoformat(str(upload_date)).date() if not isinstance(
            upload_date, (date, datetime)) else upload_date
        days = (date.today() - (d if isinstance(d, date) else d.date())).days
        return round(math.exp(-days / 180), 4)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Markdown table parser  (with space-separated fallback)
# ---------------------------------------------------------------------------

def _is_pipe_table(md: str) -> bool:
    """Return True if the string looks like a markdown pipe table."""
    return any(line.strip().startswith("|") for line in md.splitlines())


def _parse_pipe_table(md: str) -> dict:
    """Parse a GitHub-flavored markdown pipe table."""
    lines = [l.rstrip() for l in md.strip().splitlines() if l.strip()]
    if not lines:
        return _empty_parse()

    def _split(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    sep_re  = re.compile(r'^[\|\s\-:]+$')
    sep_idx = next((i for i, l in enumerate(lines) if sep_re.match(l) and i > 0), None)

    if sep_idx:
        header_row = _split(lines[0])
        data_rows  = [_split(l) for l in lines[sep_idx + 1:] if l.strip()]
        has_header = True
    else:
        header_row = []
        data_rows  = [_split(l) for l in lines]
        has_header = False

    return _build_parse_result(header_row, data_rows, has_header, md)


def _parse_space_table(md: str) -> dict:
    """
    Fallback for tables that unstructured returns as raw space-separated text
    (e.g. 'Model BLEU EN-DE EN-FR ...' without pipe characters).
    We treat the whole string as a single row for metadata purposes — the
    full text is still stored and embedded as-is.
    """
    lines     = [l.strip() for l in md.strip().splitlines() if l.strip()]
    data_rows = [l.split() for l in lines]
    return _build_parse_result([], data_rows, False, md)


def _empty_parse() -> dict:
    return {
        "header_names": [], "has_header": False, "rows": [],
        "row_count": 0, "col_count": 0,
        "has_numbers": False, "column_types": [],
    }


def _build_parse_result(
    header_row: list[str],
    data_rows:  list[list[str]],
    has_header: bool,
    md:         str,
) -> dict:
    col_count  = max(
        (len(r) for r in ([header_row] if header_row else []) + data_rows),
        default=0,
    )
    numeric_re = re.compile(r'^[\d,.$%\-+()·]+$')
    col_types: list[str] = []
    for ci in range(col_count):
        vals   = [r[ci] for r in data_rows if ci < len(r) and r[ci]]
        n_num  = sum(1 for v in vals if numeric_re.match(v.replace(" ", "")))
        col_types.append(
            "numeric" if vals and n_num / len(vals) >= 0.6 else "text"
        )
    return {
        "header_names": header_row,
        "has_header":   has_header,
        "rows":         data_rows,
        "row_count":    len(data_rows),
        "col_count":    col_count,
        "has_numbers":  bool(re.search(r'\d', md)),
        "column_types": col_types,
    }


def _parse_markdown_table(md: str) -> dict:
    """Route to pipe parser or space-separated fallback."""
    if _is_pipe_table(md):
        return _parse_pipe_table(md)
    return _parse_space_table(md)


def _rebuild_markdown(header_names: list[str], rows: list[list[str]]) -> str:
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
# LLM summary via qwen3.6
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """\
Summarize the following table in 2-3 sentences:
- What the table represents (subject/topic)
- Key values, trends, or comparisons
- Any notable outliers or patterns

Be concise and factual. Do not invent data not in the table.

TABLE:
{table_markdown}
"""


def _ollama_summarize(table_markdown: str, model: str = DEFAULT_MODEL) -> str:
    logger.info("Requesting LLM table summary via Ollama (model=%r, chars=%d)", model, len(table_markdown))
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(
                table_markdown=table_markdown[:4000]
            )}],
            options={"num_predict": 200, "temperature": 0.1},
        )
        summary = (response.get("message", {}).get("content", "") if isinstance(response, dict)
                   else response.message.content).strip()
        logger.info("LLM table summary received (%d chars)", len(summary))
        return summary
    except Exception as exc:
        logger.warning("LLM table summarization failed: %s", exc)
        return ""


def _get_summary(table_markdown: str, llm_client: Any) -> str:
    if llm_client is None:
        return ""
    if llm_client == "ollama" or (
        isinstance(llm_client, dict) and llm_client.get("backend") == "ollama"
    ):
        model = llm_client.get("model", DEFAULT_MODEL) if isinstance(llm_client, dict) else DEFAULT_MODEL
        return _ollama_summarize(table_markdown, model)
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
    chunk_id:    str,
    chunk_type:  str,
    table_index: int,
    parsed:      dict,
    llm_summary: str,
    source_type: str,
    doc_meta:    dict,
    row_start:   int | None = None,
    row_end:     int | None = None,
) -> dict:
    return {
        "chunk_id":      chunk_id,
        "doc_id":        doc_meta.get("doc_id", ""),
        "dept_id":       doc_meta.get("dept_id", ""),
        "doc_type":      source_type,
        "content_type":  "table",
        "chunk_type":    chunk_type,
        "chunk_strategy":"dual_representation",
        "table_index":   table_index,
        "row_count":     parsed["row_count"],
        "col_count":     parsed["col_count"],
        "has_header":    parsed["has_header"],
        "header_names":  parsed["header_names"],
        "column_types":  parsed["column_types"],
        "has_numbers":   parsed["has_numbers"],
        "row_start":     row_start,
        "row_end":       row_end,
        "llm_summary":   llm_summary,
        "filename":      doc_meta.get("filename", ""),
        "upload_date":   str(doc_meta.get("upload_date", "")),
        "created_by":    doc_meta.get("created_by", ""),
        "recency_score": _recency_score(doc_meta.get("upload_date")),
    }


# ---------------------------------------------------------------------------
# Dual-representation chunking
# ---------------------------------------------------------------------------

ROW_WINDOW  = 10
ROW_OVERLAP = 2
MIN_ROW_GROUP = 4   # don't create a row-group chunk with fewer than 4 rows


def _dual_chunk_table(
    md:          str,
    table_index: int,
    source_type: str,
    doc_meta:    dict,
    llm_summary: str,
) -> list[TableChunk]:
    parsed = _parse_markdown_table(md)
    chunks: list[TableChunk] = []

    # ── Full-table chunk ──────────────────────────────────────────────────────
    cid       = _make_id()
    full_text = md.strip()
    if llm_summary:
        full_text = f"Summary: {llm_summary}\n\n{full_text}"

    chunks.append(TableChunk(
        chunk_id    = cid,
        text        = full_text,
        chunk_type  = "full_table",
        llm_summary = llm_summary,
        metadata    = _base_meta(cid, "full_table", table_index, parsed,
                                 llm_summary, source_type, doc_meta,
                                 row_start=0, row_end=parsed["row_count"]),
    ))

    # ── Row-group chunks (only for tables > ROW_WINDOW rows) ─────────────────
    rows = parsed["rows"]
    if len(rows) > ROW_WINDOW:
        step = ROW_WINDOW - ROW_OVERLAP
        for start in range(0, len(rows), step):
            end         = min(start + ROW_WINDOW, len(rows))
            window_rows = rows[start:end]
            if len(window_rows) < MIN_ROW_GROUP:
                continue          # skip tiny trailing fragments
            window_md = _rebuild_markdown(parsed["header_names"], window_rows)
            if not window_md:
                continue
            cid = _make_id()
            chunks.append(TableChunk(
                chunk_id    = cid,
                text        = window_md,
                chunk_type  = "row_group",
                llm_summary = llm_summary,
                metadata    = _base_meta(cid, "row_group", table_index, parsed,
                                         llm_summary, source_type, doc_meta,
                                         row_start=start, row_end=end),
            ))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_tables(
    tables:     list[str],
    source_type: str,
    doc_meta:   dict,
    llm_client: Any = None,
) -> list[TableChunk]:
    """
    Chunk and tag tables from a parsed document.

    Args:
        tables:      Markdown table strings from ParsedDocument.tables.
                     Also handles space-separated tables from unstructured.
        source_type: "pdf" | "docx" | "pptx" | "url" | etc.
        doc_meta:    {doc_id, dept_id, filename, upload_date, created_by}
        llm_client:  LLM backend for table summarization.
                     None    → skip summaries
                     "ollama"→ qwen3.6 with default tag
                     {"backend":"ollama","model":"qwen3.6:27b"} → specific tag
                     callable→ custom function

    Returns:
        List of TableChunk objects (1 full_table + N row_group per table).
    """
    if not tables:
        logger.info("process_tables: no tables to process")
        return []

    logger.info("process_tables: processing %d table(s) (llm_client=%s)", len(tables), llm_client is not None)
    all_chunks: list[TableChunk] = []
    for idx, md in enumerate(tables):
        md = md.strip()
        if not md:
            logger.debug("Skipping empty table at index %d", idx)
            continue
        logger.info("Processing table %d/%d (%d chars)", idx + 1, len(tables), len(md))
        summary = _get_summary(md, llm_client)
        new_chunks = _dual_chunk_table(md, idx, source_type, doc_meta, summary)
        logger.info("Table %d produced %d chunks (full_table + row_groups)", idx, len(new_chunks))
        all_chunks.extend(new_chunks)

    logger.info("process_tables complete: %d total table chunks", len(all_chunks))
    return all_chunks