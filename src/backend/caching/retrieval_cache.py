# caching/retrieval_cache.py — tier 2: get-or-compute around fanout->merge->rerank->recency.
import json
import uuid

from caching.cache_keys import retrieval_key
from core.config import CACHE_TTL_RETRIEVAL
from core.valkey_client import get_client


class CachedPoint:
    """Minimal stand-in for a Qdrant ScoredPoint. Only .payload is read
    downstream (generation.py, confidence_gate.py, recency.py), so a full
    re-hydration into the real qdrant_client type isn't needed."""

    def __init__(self, payload: dict):
        self.payload = payload


def _serialize(scored_points: list[tuple[object, float]]) -> str:
    return json.dumps([{"score": score, "payload": p.payload} for p, score in scored_points])


def _deserialize(raw: str) -> list[tuple[object, float]]:
    return [(CachedPoint(item["payload"]), item["score"]) for item in json.loads(raw)]


async def get_cached(
    resolved_query: str, dept_ids: list[uuid.UUID], versions: dict[uuid.UUID, int]
) -> list[tuple[object, float]] | None:
    raw = await get_client().get(retrieval_key(resolved_query, dept_ids, versions))
    return _deserialize(raw) if raw else None


async def set_cached(
    resolved_query: str,
    dept_ids: list[uuid.UUID],
    versions: dict[uuid.UUID, int],
    scored_points: list[tuple[object, float]],
) -> None:
    key = retrieval_key(resolved_query, dept_ids, versions)
    await get_client().set(key, _serialize(scored_points), ex=CACHE_TTL_RETRIEVAL)


if __name__ == "__main__":
    points = [(CachedPoint({"chunk_id": "c1", "text": "hi"}), 0.8)]
    raw = _serialize(points)
    restored = _deserialize(raw)
    assert restored[0][0].payload == points[0][0].payload
    assert restored[0][1] == points[0][1]
    print("caching/retrieval_cache.py: self-check passed")
