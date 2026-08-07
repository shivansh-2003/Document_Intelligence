# indexing/indexing_pipeline.py — chunk -> embed -> vectorstore write -> corpus version bump
#
# Deliberately separate from parsing/ and pipelines/: those stages produce chunks and
# stop there. Nothing in them calls embedding_service or vector_store -- indexing is
# always a later, distinct step, called from the API layer once chunks exist.
import uuid

from core.database import SessionLocal
from ingestion_versioning.corpus_version import bump_corpus_version
from ingestion_versioning.supersession import detect_duplicates
from pipelines.text_pipeline import Chunk
from vectorstore.vector_store import upsert_chunks


async def index_chunks(chunks: list[Chunk]) -> list[str]:
    """Graph enqueueing (structural edges always, entity extraction for text/audio/
    diagram chunks per vector-graph-routing.md) lands here once graph/ exists --
    for now this is vector-store indexing, the corpus_version bump (§7.1) that
    invalidates stale retrieval/response cache entries for that department, and
    supersession candidate detection (§7.2). Returns possible-duplicate doc_ids --
    informational only, callers surface it, nothing here acts on it.

    Opens its own session rather than depending on a caller-supplied one -- same
    reasoning as api/parsing_router.py's _mark_document: callers here (the SSE
    generator, the batch background task) can outlive the request that triggered
    them.
    """
    if not chunks:
        return []
    await upsert_chunks(chunks)
    dept_id, doc_id = chunks[0].metadata.dept_id, chunks[0].metadata.doc_id
    async with SessionLocal() as db:
        await bump_corpus_version(db, uuid.UUID(dept_id))
    return await detect_duplicates(dept_id, doc_id, chunks)
