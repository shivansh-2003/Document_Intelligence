# table_pipeline.py
import logging

from services.llm_service import generate
from utils.parsing_utils import RawElement

logger = logging.getLogger(__name__)

TABLE_PROMPT_TEMPLATE = """You are summarizing a table for a RAG retrieval system. Someone will search for
this table using natural language — your summary is the only thing that gets embedded and matched, the
original table stays available separately for exact lookup.

Write one dense paragraph, in order:
1. Subject — what the table is about, inferred from headers and cell content — one sentence.
2. Structure — row and column count, what each column represents, and the unit each numeric column is
   in (currency, %, count, date, ...) if shown.
3. Standout values — the highest, lowest, first, last, or otherwise notable figures, each named
   explicitly with its row/column label attached (e.g. "Platform revenue reached $157,166K in 2024, up
   from $147,509K in 2023" — not "revenue increased"). For a non-numeric table (categories, yes/no,
   status flags), name the distinguishing rows/columns instead — what's present in one row but not
   another.
4. Trend or comparison the numbers show, only if the data actually supports it — don't invent one.

Rules:
- Every number, label, and unit you cite must appear in the table exactly as shown — same currency
  symbol, same decimal precision, same casing. Do not compute, round, convert, or infer a value that
  isn't printed.
- Multi-level or merged headers: resolve each column to its full path (e.g. "2024 > Q1 > Revenue"),
  don't take the bottom-most header cell in isolation.
- No preamble ("This table shows..."), no markdown, no bullet points. Plain prose only.
- If the table is mostly blank/structural (e.g. a form template) or has no interpretable values, say so
  plainly in one sentence instead of padding.
- 8 sentences max regardless of table size — for a large table, prioritize the most extreme/notable rows
  over exhaustive coverage.

Table HTML:
{html}"""

def build_table_prompt(html: str) -> str:
    return TABLE_PROMPT_TEMPLATE.format(html=html)
    
def describe_table(el: RawElement) -> RawElement:
    if el.type != "Table" or not el.metadata.text_as_html:
        return el
    logger.info("Describing table element_id=%s", el.element_id)
    el.text = generate(build_table_prompt(el.metadata.text_as_html))   # overwrite, html stays as source of truth
    return el


def describe_tables(elements: list[RawElement]) -> list[RawElement]:
    tables = [e for e in elements if e.type == "Table" and e.metadata.text_as_html]
    logger.info("Describing %d table(s) of %d elements", len(tables), len(elements))
    return [describe_table(e) for e in elements]