from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image, CompositeElement, Text


@dataclass
class ParsedDocument:
    texts: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


class PDFParser:
    def __init__(self, extract_images: bool = True, strategy: str = "hi_res"):
        self.extract_images = extract_images
        self.strategy = strategy
        self._result: Optional[ParsedDocument] = None

    def parse(self, file_path: str | Path) -> ParsedDocument:
        file_path = Path(file_path)
        elements = partition_pdf(
            filename=str(file_path),
            strategy=self.strategy,
            extract_images_in_pdf=self.extract_images,
        )
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
