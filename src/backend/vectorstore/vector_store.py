# vectorstore/vector_store.py — Qdrant collection mgmt, hybrid upsert/query
import asyncio
import logging
import uuid

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from core.config import QDRANT_API_KEY, QDRANT_URL
from indexing.embedding_service import DENSE_SIZE, chunk_embed_text, embed_batch
from pipelines.text_pipeline import Chunk

logger = logging.getLogger(__name__)

COLLECTION = "chunks"
UPSERT_RETRIES = 3

# Default AsyncQdrantClient timeout is a flat 5s across connect/read/write (verified:
# Timeout(timeout=5.0)) -- too tight for a burst of concurrent fresh connections to a
# cloud endpoint (observed directly: 2 of 4 concurrent /parse/batch uploads hit
# ConnectTimeout at the default). 30s gives connection setup real headroom.
_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
_ensured = False


async def ensure_collection() -> None:
    """Idempotent, called lazily on first use -- importing this module must not
    require Qdrant to already be reachable."""
    global _ensured
    if _ensured:
        return
    if not await _client.collection_exists(COLLECTION):
        await _client.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        # NON-NEGOTIABLE: dept_id is the tenant boundary. Indexing it makes it a
        # pre-filter -- Qdrant excludes non-matching points before the ANN search
        # runs, not after. Without this every query is an unscoped full-collection
        # scan across every department.
        await _client.create_payload_index(COLLECTION, "dept_id", models.PayloadSchemaType.KEYWORD)
    _ensured = True


def _point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be an unsigned int or a UUID -- an arbitrary prefixed
    string like "chnk_a1b2c3d4e5f6" is rejected outright. uuid5 derives a UUID
    deterministically from chunk_id, so the same chunk always maps to the same point
    (re-processing a document overwrites, doesn't duplicate) -- chunk_id itself still
    rides in the payload for the actual cross-system correlation."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _to_payload(c: Chunk) -> dict:
    m = c.metadata
    return {
        "dept_id": m.dept_id,
        "doc_id": m.doc_id,
        "doc_type": m.doc_type,
        "chunk_id": m.chunk_id,
        "kind": c.kind,
        "title": c.title,
        "filename": m.filename,
        "page_number": m.page_number,
        "text": c.text,
    }


async def upsert_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return
    await ensure_collection()

    texts = [chunk_embed_text(c) for c in chunks]
    # embed_batch is synchronous, CPU-bound ONNX inference -- calling it directly here
    # would block the entire event loop for the whole embedding run (observed directly:
    # a 91-chunk batch stalled ~2m43s between "chunks ready" and "upserting", during
    # which the WHOLE server -- not just this task -- couldn't make progress on
    # anything else). run_in_executor is the same fix already applied to
    # parse_document() and faster-whisper (utils/async_utils.iter_in_thread).
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(None, embed_batch, texts)

    points = [
        models.PointStruct(
            id=_point_id(c.metadata.chunk_id),
            vector={
                "dense": v["dense"],
                "sparse": models.SparseVector(indices=v["sparse"]["indices"], values=v["sparse"]["values"]),
            },
            payload=_to_payload(c),
        )
        for c, v in zip(chunks, vectors)
    ]
    logger.info("Upserting %d point(s) into %r", len(points), COLLECTION)
    await _upsert_with_retry(points)


async def _upsert_with_retry(points: list) -> None:
    """A transient network blip to Qdrant Cloud shouldn't permanently drop a document
    from the index -- observed directly, not speculative: under concurrent /parse/batch
    load, some upserts hit ConnectTimeout while sibling requests in the same batch
    succeeded. ponytail: fixed attempt count + exponential backoff, in-process --
    doesn't survive a server restart mid-retry. Real durability is Phase 5 (Celery
    retry/backoff on a persistent queue), not this."""
    for attempt in range(1, UPSERT_RETRIES + 1):
        try:
            await _client.upsert(collection_name=COLLECTION, points=points)
            return
        except ResponseHandlingException:
            if attempt == UPSERT_RETRIES:
                raise
            wait = 2**attempt
            logger.warning(
                "Qdrant upsert attempt %d/%d failed, retrying in %ds",
                attempt, UPSERT_RETRIES, wait, exc_info=True,
            )
            await asyncio.sleep(wait)


async def hybrid_search(query: str, dept_id: str, limit: int = 10, doc_id: str | None = None) -> list:
    """dept_id is mandatory, not a filter you remember to add -- an empty string or
    None here would otherwise produce Filter(must=[]), a filter that filters nothing,
    turning a coding bug into a cross-tenant data leak instead of a loud failure."""
    if not dept_id:
        raise ValueError("dept_id is required -- refusing an unscoped search")

    await ensure_collection()

    must = [models.FieldCondition(key="dept_id", match=models.MatchValue(value=dept_id))]
    if doc_id:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))

    loop = asyncio.get_running_loop()
    v = (await loop.run_in_executor(None, embed_batch, [query]))[0]
    result = await _client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=v["dense"], using="dense", limit=limit * 4),
            models.Prefetch(
                query=models.SparseVector(indices=v["sparse"]["indices"], values=v["sparse"]["values"]),
                using="sparse",
                limit=limit * 4,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=models.Filter(must=must),
        limit=limit,
    )
    return result.points
