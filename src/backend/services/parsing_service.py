# parsing_service.py
import logging
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from parsing.document_parser import DocumentPartitioner
from pipelines.image_pipeline import describe_images
from pipelines.table_pipeline import describe_tables
from pipelines.text_pipeline import Chunk, normalize_document
from utils.parsing_utils import insight

logger = logging.getLogger(__name__)

# Suffix -> doc_type. Aliases normalise so downstream never sees both "htm" and "html".
DOC_TYPES = {
    ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx",
    ".txt": "txt",
    ".md": "md", ".markdown": "md",
    ".html": "html", ".htm": "html",
}


class ParsedDocument(BaseModel):
    doc_id: str
    dept_id: str
    source: str
    doc_type: str
    chunks: list[Chunk]
    counts: dict[str, int]


_partitioner = DocumentPartitioner()


def parse_document(
    path: str,
    doc_id: str,
    dept_id: str,
    source_name: str | None = None,
) -> ParsedDocument:
    """file -> chunks, fully tagged and ready to embed.

    doc_id and dept_id are REQUIRED, not optional. dept_id is the multi-tenant
    boundary — it selects the Qdrant collection and the Neo4j label. A chunk that
    reaches the indexer without one is either unroutable or, worse, routable to the
    wrong tenant. Fail here, at the trust boundary, rather than trusting every future
    call site to remember to bolt it on.
    """
    if not doc_id or not doc_id.strip():
        raise ValueError("doc_id is required")
    if not dept_id or not dept_id.strip():
        raise ValueError("dept_id is required")

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    doc_type = DOC_TYPES.get(p.suffix.lower())
    if doc_type is None:
        raise ValueError(f"unsupported: {p.suffix.lower()}")

    source = source_name or p.name
    logger.info("Parsing %r doc_id=%s dept_id=%s doc_type=%s", source, doc_id, dept_id, doc_type)

    elements = _partitioner.partition(str(p))
    for e in elements:
        e.metadata.filename = source

    counts = dict(insight(elements))   # raw category counts, taken before filtering/grouping

    describe_tables(elements)          # Table.text -> LLM summary  (md/html tables included)
    describe_images(elements)          # Image.text -> LLM JSON string (no-ops without image bytes)

    chunks = normalize_document(elements)   # filter -> captions -> title grouping -> size cap
    _tag(chunks, doc_id=doc_id, dept_id=dept_id, doc_type=doc_type)

    logger.info("Parsed %r into %d chunks %s", source, len(chunks), counts)
    return ParsedDocument(
        doc_id=doc_id, dept_id=dept_id, source=source,
        doc_type=doc_type, chunks=chunks, counts=counts,
    )


def _tag(chunks: list[Chunk], doc_id: str, dept_id: str, doc_type: str) -> None:
    """Stamp document-scoped identity onto every chunk. In-place; runs once.

    MUST run after normalize_document, never inside group_by_title. split_oversized
    (the last step of normalize_document) clones a chunk's metadata verbatim when it
    splits an oversized text chunk — a chunk_id minted before the split would be
    duplicated across every sub-chunk, and Qdrant, which upserts by id, would keep
    only the last one and silently drop the rest.
    """
    for c in chunks:
        c.metadata.chunk_id = "chnk_" + uuid4().hex[:12]
        c.metadata.doc_id = doc_id
        c.metadata.dept_id = dept_id
        c.metadata.doc_type = doc_type