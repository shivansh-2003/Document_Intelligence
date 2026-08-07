# caching/response_cache.py — tier 3: cosine match against stored query embeddings.
#
# Not an exact key match -- SCAN the scope's small candidate set and brute-force
# cosine-compare in Python. A scope's candidate set is a handful of recent
# queries, not a corpus; reusing Qdrant for this would be reaching for a search
# index to solve a cache problem. See context/retrieval.md §8.
import json
import math
import uuid

from caching.cache_keys import response_scope_prefix
from caching.embedding_cache import get_or_embed
from core.config import CACHE_TTL_RESPONSE, RESPONSE_CACHE_SIM_THRESHOLD
from core.valkey_client import get_client


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def get_cached(resolved_query: str, dept_ids: list[uuid.UUID], versions: dict[uuid.UUID, int]) -> dict | None:
    client = get_client()
    query_vector = (await get_or_embed(resolved_query))["dense"]
    prefix = response_scope_prefix(dept_ids, versions)

    best_score, best_entry = 0.0, None
    async for key in client.scan_iter(match=f"{prefix}*"):
        raw = await client.get(key)
        if not raw:
            continue
        entry = json.loads(raw)
        score = _cosine(query_vector, entry["query_vector"])
        if score > best_score:
            best_score, best_entry = score, entry

    return best_entry if best_entry and best_score >= RESPONSE_CACHE_SIM_THRESHOLD else None


async def set_cached(
    resolved_query: str,
    dept_ids: list[uuid.UUID],
    versions: dict[uuid.UUID, int],
    answer: str,
    citations: list[dict],
) -> None:
    client = get_client()
    query_vector = (await get_or_embed(resolved_query))["dense"]
    prefix = response_scope_prefix(dept_ids, versions)
    key = f"{prefix}{uuid.uuid4().hex[:12]}"
    entry = {"query_vector": query_vector, "answer": answer, "citations": citations}
    await client.set(key, json.dumps(entry), ex=CACHE_TTL_RESPONSE)


if __name__ == "__main__":
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], []) == 0.0
    print("caching/response_cache.py: self-check passed")
