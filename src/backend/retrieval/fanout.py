# retrieval/fanout.py — §5.1: asyncio.gather over (sub_query x dept_id) variants -> hybrid_search
import asyncio
import uuid

from caching.embedding_cache import get_or_embed
from core.config import FANOUT_TOP_K
from retrieval.query_transform import SubQuery
from vectorstore.vector_store import hybrid_search

VariantKey = tuple[str, str]  # (sub_query.text, dept_id)


async def fanout(sub_queries: list[SubQuery], dept_ids: list[uuid.UUID]) -> dict[VariantKey, list]:
    """Every (sub_query, dept_id) pair searched concurrently. Keyed by
    (sub_query.text, dept_id) so merge.py can attribute each result back to its
    department for the quota pass -- this is orchestration only, hybrid_search
    (dense+sparse fusion, dept_id pre-filter) does the actual retrieval work.

    Each unique sub_query text is embedded once (tier-1 cached) and reused
    across every department it fans out to, rather than re-embedding per pair.
    """
    unique_texts = {sq.text for sq in sub_queries}
    vectors = dict(zip(unique_texts, await asyncio.gather(*(get_or_embed(t) for t in unique_texts))))

    variants = [(sq, dept_id) for sq in sub_queries for dept_id in dept_ids]

    async def _one(sq: SubQuery, dept_id: uuid.UUID) -> tuple[VariantKey, list]:
        points = await hybrid_search(sq.text, str(dept_id), limit=FANOUT_TOP_K, vector=vectors[sq.text])
        return (sq.text, str(dept_id)), points

    results = await asyncio.gather(*(_one(sq, dept_id) for sq, dept_id in variants))
    return dict(results)
