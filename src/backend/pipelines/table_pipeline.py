# table_pipeline.py
import logging

from services.llm_service import generate
from utils.parsing_utils import RawElement

logger = logging.getLogger(__name__)

TABLE_PROMPT_TEMPLATE = """You are summarizing a table for a RAG retrieval system. Someone will search for
this table using natural language — your summary is the only thing that gets embedded and matched.

Write a dense paragraph that includes, in order:
1. What the table is about (its subject, inferred from headers/captions/context) — one sentence.
2. Structure: row and column count, what each column represents.
3. The specific standout values — the highest, lowest, first, last, or otherwise notable figures,
   named explicitly with their row/column label attached (e.g. "Platform revenue reached $157,166K
   in 2024, up from $147,509K in 2023" — not "revenue increased").
4. Any trend or comparison the numbers show, only if the data actually supports it — don't invent one.

Rules:
- Every number you cite must appear in the table exactly as shown. Do not compute, round, or infer values.
- No preamble ("This table shows..."), no markdown, no bullet points. Plain prose only.
- If the table is mostly blank/structural (e.g. a form template), say so plainly instead of padding.

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