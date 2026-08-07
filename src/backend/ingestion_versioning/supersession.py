# ingestion_versioning/supersession.py — §7.2: nearest-neighbor candidate detection.
#
# Detection is automatic and informational only -- runs right after ingest and
# surfaces candidate doc_ids to the caller. It never flips anything itself;
# confirming supersession is a separate, explicit, editor-gated route
# (api/documents_router.py) that calls vector_store.mark_superseded. Silently
# hiding a document from retrieval is a hard-to-reverse action -- it needs an
# explicit confirming call, not a heuristic. See context/retrieval.md §7.2.
import asyncio
from collections import Counter

from indexing.embedding_service import embed_batch
from pipelines.text_pipeline import Chunk
from vectorstore.vector_store import dense_search

SIM_THRESHOLD = 0.92     # cosine -- deliberately high, this only flags candidates
SAMPLE_SIZE = 5          # chunks sampled per document; best-effort, not exhaustive
MIN_MATCHING_CHUNKS = 2  # one coincidental match isn't a duplicate signal


async def detect_duplicates(dept_id: str, doc_id: str, chunks: list[Chunk]) -> list[str]:
    """New document's chunks -> distinct doc_ids in the same department that look
    like a near-duplicate or prior version. Samples rather than checking every
    chunk -- this is a post-ingest signal, not a correctness gate."""
    sample = [c for c in chunks if c.kind == "text"][:SAMPLE_SIZE] or chunks[:SAMPLE_SIZE]
    if not sample:
        return []

    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(None, embed_batch, [c.text for c in sample])

    matches: Counter = Counter()
    for v in vectors:
        for p in await dense_search(v["dense"], dept_id, limit=3, exclude_doc_id=doc_id):
            if p.score >= SIM_THRESHOLD:
                matches[p.payload["doc_id"]] += 1

    return [candidate for candidate, count in matches.items() if count >= MIN_MATCHING_CHUNKS]
