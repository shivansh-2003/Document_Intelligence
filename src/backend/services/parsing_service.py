# services/parsing_service.py
from pathlib import Path

from pydantic import BaseModel

from parsing.document_parser import DocumentPartitioner
from utils.parsing_utils import RawElement, insight


class ParsedDocument(BaseModel):
    source: str
    doc_type: str                 # pdf | docx | pptx
    elements: list[RawElement]    # ordered, raw, nothing dropped
    counts: dict[str, int]        # {'Title': 208, 'Table': 72, 'Image': 3, ...}


_partitioner = DocumentPartitioner()


def parse_document(path: str) -> ParsedDocument:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    elements = _partitioner.partition(str(p))
    return ParsedDocument(
        source=p.name,
        doc_type=p.suffix.lower().lstrip("."),
        elements=elements,
        counts=dict(insight(elements)),
    )