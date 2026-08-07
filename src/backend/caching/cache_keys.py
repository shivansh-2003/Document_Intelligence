# caching/cache_keys.py — builds tier 1/2/3 keys. See context/retrieval.md §8.
#
# Tier 2/3 keys bake in each scoped department's corpus_version -- a re-ingest
# bumps that counter (ingestion_versioning/corpus_version.py) and old keys
# naturally miss, no separate cache-bust step needed.
import hashlib
import uuid


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _scope_key(dept_ids: list[uuid.UUID], versions: dict[uuid.UUID, int]) -> str:
    return ",".join(f"{d}:{versions.get(d, 0)}" for d in sorted(dept_ids))


def embedding_key(text: str) -> str:
    return f"emb:{_hash(text)}"


def retrieval_key(resolved_query: str, dept_ids: list[uuid.UUID], versions: dict[uuid.UUID, int]) -> str:
    return f"ret:{_scope_key(dept_ids, versions)}:{_hash(resolved_query)}"


def response_scope_prefix(dept_ids: list[uuid.UUID], versions: dict[uuid.UUID, int]) -> str:
    """Prefix scanned by response_cache.py's cosine-match lookup -- not a full
    key. The query text never determines the tier-3 key by itself; that's the
    point of a similarity cache rather than an exact-match one."""
    return f"resp:{_scope_key(dept_ids, versions)}:"


if __name__ == "__main__":
    a, b = uuid.uuid4(), uuid.uuid4()
    v1 = {a: 1, b: 2}
    v2 = {a: 1, b: 3}  # b's corpus changed
    assert retrieval_key("q", [a, b], v1) != retrieval_key("q", [a, b], v2)
    assert retrieval_key("q", [a, b], v1) == retrieval_key("q", [b, a], v1)  # order-independent
    assert embedding_key("same") == embedding_key("same")
    print("caching/cache_keys.py: self-check passed")
