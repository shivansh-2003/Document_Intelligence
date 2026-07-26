# indexing/indexing_pipeline.py — chunk -> embed -> vectorstore write
#
# Deliberately separate from parsing/ and pipelines/: those stages produce chunks and
# stop there. Nothing in them calls embedding_service or vector_store -- indexing is
# always a later, distinct step, called from the API layer once chunks exist.
from pipelines.text_pipeline import Chunk
from vectorstore.vector_store import upsert_chunks


async def index_chunks(chunks: list[Chunk]) -> None:
    """Graph enqueueing (structural edges always, entity extraction for text/audio/
    diagram chunks per vector-graph-routing.md) lands here once graph/ exists --
    for now this is vector-store indexing only."""
    await upsert_chunks(chunks)
