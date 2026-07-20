# parsing_service.py
import logging
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from pydantic import BaseModel

from parsing.audio_parser import transcribe_stream
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
AUDIO_TYPES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


class ParsedDocument(BaseModel):
    doc_id: str
    dept_id: str
    source: str
    doc_type: str
    chunks: list[Chunk]
    counts: dict[str, int]


_partitioner = DocumentPartitioner()


def _check_identity(doc_id: str, dept_id: str) -> None:
    """dept_id is the multi-tenant boundary -- it selects the Qdrant collection and
    the Neo4j label. A chunk that reaches the indexer without one is either
    unroutable or, worse, routable to the wrong tenant. Fail here, at the trust
    boundary, rather than trusting every future call site to remember it."""
    if not doc_id or not doc_id.strip():
        raise ValueError("doc_id is required")
    if not dept_id or not dept_id.strip():
        raise ValueError("dept_id is required")


def _tag_chunk(c: Chunk, doc_id: str, dept_id: str, doc_type: str) -> Chunk:
    c.metadata.chunk_id = "chnk_" + uuid4().hex[:12]
    c.metadata.doc_id = doc_id
    c.metadata.dept_id = dept_id
    c.metadata.doc_type = doc_type
    return c


def parse_document(
    path: str,
    doc_id: str,
    dept_id: str,
    source_name: str | None = None,
) -> ParsedDocument:
    """file (pdf/docx/pptx/txt/md/html) -> chunks, fully tagged and ready to embed."""
    _check_identity(doc_id, dept_id)

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

    # normalize_document ends with split_oversized, which clones metadata verbatim
    # across sub-chunks -- so identity is tagged AFTER, once per final chunk, not
    # before. Tagging before the split would duplicate chunk_id across sub-chunks,
    # and Qdrant (upsert by id) would silently keep only the last one.
    chunks = normalize_document(elements)
    for c in chunks:
        _tag_chunk(c, doc_id, dept_id, doc_type)

    logger.info("Parsed %r into %d chunks %s", source, len(chunks), counts)
    return ParsedDocument(
        doc_id=doc_id, dept_id=dept_id, source=source,
        doc_type=doc_type, chunks=chunks, counts=counts,
    )


async def parse_audio_stream(
    path: str,
    doc_id: str,
    dept_id: str,
    source_name: str | None = None,
) -> AsyncIterator[Chunk]:
    """audio -> Chunks, yielded as Whisper segments transcribe.

    Each chunk already carries its own start_sec/end_sec (assigned per-segment in
    audio_parser, never cloned by a later split step), so tagging one at a time as
    they stream in is safe -- unlike the file path, there is no later step that
    could duplicate a chunk_id across siblings.
    """
    _check_identity(doc_id, dept_id)

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    if p.suffix.lower() not in AUDIO_TYPES:
        raise ValueError(f"unsupported audio type: {p.suffix.lower()}")

    source = source_name or p.name
    logger.info("Streaming transcription %r doc_id=%s dept_id=%s", source, doc_id, dept_id)

    n = 0
    async for chunk in transcribe_stream(path, filename=source):
        n += 1
        yield _tag_chunk(chunk, doc_id, dept_id, "audio")
    logger.info("Streamed %r into %d chunks", source, n)