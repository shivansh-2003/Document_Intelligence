# table_pipeline.py
import logging

from services.llm_service import generate
from utils.parsing_utils import RawElement

logger = logging.getLogger(__name__)

TABLE_PROMPT_TEMPLATE = """Summarize this HTML table for retrieval in a RAG system.
State row/column count, what the columns represent, and the key figures or trend —
not a cell-by-cell transcript. No preamble.

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