from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from unstructured.partition.text import partition_text
from unstructured.documents.elements import Table, Image, CompositeElement, Text


@dataclass
class ParsedDocument:
    texts: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


class TextParser:
    def __init__(self):
        self._result: Optional[ParsedDocument] = None

    def parse(self, file_path: str | Path) -> ParsedDocument:
        file_path = Path(file_path)
        elements = partition_text(filename=str(file_path))
        self._result = self._partition_elements(elements)
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
