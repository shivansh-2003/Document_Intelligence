# parsing/document_parser.py
import logging
import re
from pathlib import Path

from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text

from utils.parsing_utils import RawElement, to_raw

logger = logging.getLogger(__name__)

FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


class DocumentPartitioner:
    """Raw partition only. Order preserved, nothing collapsed or dropped."""

    def partition(self, path: str) -> list[RawElement]:
        suffix = Path(path).suffix.lower()
        fn = {
            ".pdf": self.pdf,
            ".docx": self.docx,
            ".pptx": self.pptx,
            ".txt": self.text,
            ".md": self.md,
            ".markdown": self.md,
            ".html": self.html,
            ".htm": self.html,
        }.get(suffix)
        if fn is None:
            logger.error("Unsupported file type %r for %r", suffix, path)
            raise ValueError(f"unsupported: {suffix}")
        logger.info("Partitioning %r as %s", path, suffix)
        elements = fn(path)
        logger.info("Partitioned %r into %d elements", path, len(elements))
        return elements

    def pdf(self, path: str) -> list[RawElement]:
        els = partition_pdf(
            filename=path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )
        return to_raw(els)

    def docx(self, path: str) -> list[RawElement]:
        return to_raw(partition_docx(filename=path, infer_table_structure=True))

    def pptx(self, path: str) -> list[RawElement]:
        return to_raw(partition_pptx(filename=path, infer_table_structure=True))

    # ── paginationless text sources ─────────────────────────────────────────
    # No page_number, no image bytes. Tables (md/html) still arrive with
    # text_as_html, so table_pipeline works on them unchanged.

    def text(self, path: str) -> list[RawElement]:
        # Explicit utf-8: unstructured's chardet autodetect misfires on short files.
        return to_raw(partition_text(filename=path, encoding="utf-8"))

    def md(self, path: str) -> list[RawElement]:
        # partition_md renders markdown -> HTML -> partition_html internally,
        # so headings become Title and pipe tables become Table + text_as_html.
        raw = Path(path).read_text(encoding="utf-8")
        return to_raw(partition_md(text=FRONTMATTER.sub("", raw)))

    def html(self, path: str) -> list[RawElement]:
        return to_raw(partition_html(filename=path))