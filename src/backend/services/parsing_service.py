# parsing_service.py
import logging
from pathlib import Path
from pydantic import BaseModel

from parsing.document_parser import DocumentPartitioner
from utils.parsing_utils import RawElement, insight
from pipelines.table_pipeline import describe_tables
from pipelines.image_pipeline import describe_images

logger = logging.getLogger(__name__)


class ParsedDocument(BaseModel):
    source: str
    doc_type: str
    elements: list[RawElement]
    counts: dict[str, int]


_partitioner = DocumentPartitioner()


def parse_document(path: str, source_name: str | None = None) -> ParsedDocument:
    p = Path(path)
    if not p.is_file():
        logger.error("File not found: %r", path)
        raise FileNotFoundError(path)

    name = source_name or p.name
    logger.info("Parsing document %r (source=%r)", str(p), name)
    elements = _partitioner.partition(str(p))
    for e in elements:
        e.metadata.filename = name

    describe_tables(elements)      # Table.text: plain summary string
    describe_images(elements)      # Image.text: JSON string {"type", "description"}

    counts = dict(insight(elements))
    logger.info("Finished parsing %r: %s", name, counts)
    return ParsedDocument(source=name, doc_type=p.suffix.lower().lstrip("."), elements=elements, counts=counts)