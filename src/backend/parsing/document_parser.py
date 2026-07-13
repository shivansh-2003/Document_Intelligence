# parsing/document_parser.py
import logging
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx

from utils.parsing_utils import RawElement, to_raw

logger = logging.getLogger(__name__)


class DocumentPartitioner:
    """Raw partition only. Order preserved, nothing collapsed or dropped."""

    def partition(self, path: str) -> list[RawElement]:
        suffix = Path(path).suffix.lower()
        fn = {".pdf": self.pdf, ".docx": self.docx, ".pptx": self.pptx}.get(suffix)
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
        return to_raw(els, path)

    def docx(self, path: str) -> list[RawElement]:
        return to_raw(partition_docx(filename=path, infer_table_structure=True), path)

    def pptx(self, path: str) -> list[RawElement]:
        return to_raw(partition_pptx(filename=path, infer_table_structure=True), path)