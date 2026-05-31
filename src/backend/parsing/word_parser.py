import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import Table, Image, CompositeElement, Text

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    texts: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


class WordParser:
    def __init__(self):
        self._result: Optional[ParsedDocument] = None

    def parse(self, file_path: str | Path) -> ParsedDocument:
        file_path = Path(file_path)
        logger.info("Partitioning DOCX: %s", file_path.name)
        elements = partition_docx(filename=str(file_path))
        logger.info("DOCX partitioned into %d raw elements", len(elements))
        self._result = self._partition_elements(elements)
        logger.info("DOCX parse complete: %d texts, %d tables, %d images",
                    len(self._result.texts), len(self._result.tables), len(self._result.images))
        return self._result

    def _partition_elements(self, elements: list) -> ParsedDocument:
        doc = ParsedDocument()
        for el in elements:
            if isinstance(el, Table):
                doc.tables.append(el.text)
            elif isinstance(el, Image):
                doc.images.append({"metadata": el.metadata.to_dict(), "text": el.text})
            elif isinstance(el, (Text, CompositeElement)):
                if el.text.strip():
                    doc.texts.append(el.text)
        return doc

    @property
    def result(self) -> Optional[ParsedDocument]:
        return self._result
