# parsing_service.py
from pathlib import Path
from pydantic import BaseModel

from parsing.document_parser import DocumentPartitioner
from utils.parsing_utils import insight
from pipelines.table_pipeline import describe_tables
from pipelines.image_pipeline import describe_images
from pipelines.text_pipeline import normalize_document, Chunk


class ParsedDocument(BaseModel):
    source: str
    doc_type: str
    chunks: list[Chunk]
    counts: dict[str, int]


_partitioner = DocumentPartitioner()


def parse_document(path: str, source_name: str | None = None) -> ParsedDocument:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    name = source_name or p.name
    elements = _partitioner.partition(str(p))
    for e in elements:
        e.metadata.filename = name

    counts = dict(insight(elements))   # raw category counts, taken before filtering/grouping

    describe_tables(elements)          # Table.text -> LLM summary
    describe_images(elements)          # Image.text -> LLM JSON string
    chunks = normalize_document(elements)   # filter -> caption attach -> title grouping

    return ParsedDocument(source=name, doc_type=p.suffix.lower().lstrip("."), chunks=chunks, counts=counts)