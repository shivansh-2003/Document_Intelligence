# caching/embedding_cache.py — tier 1: query embedding get-or-compute.
import asyncio
import json

from caching.cache_keys import embedding_key
from core.config import CACHE_TTL_EMBEDDING
from core.valkey_client import get_client
from indexing.embedding_service import embed_batch


async def get_or_embed(query: str) -> dict:
    """query -> {"dense": [...], "sparse": {...}}, cached by exact query text.
    Feeds vectorstore.vector_store.hybrid_search()'s vector= param so a cache
    hit skips the ONNX embed call entirely -- see context/retrieval.md §8."""
    client = get_client()
    key = embedding_key(query)
    cached = await client.get(key)
    if cached:
        return json.loads(cached)

    loop = asyncio.get_running_loop()
    vector = (await loop.run_in_executor(None, embed_batch, [query]))[0]
    await client.set(key, json.dumps(vector), ex=CACHE_TTL_EMBEDDING)
    return vector
