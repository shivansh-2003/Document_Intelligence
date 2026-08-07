# retrieval/merge.py — §4.2/§5.3: two-level RRF, dedupe by chunk_id, per-dept quota
#
# Level 1 (dense+sparse fusion) already happened inside hybrid_search's Qdrant
# FusionQuery. This is level 2: fuse across sub_query/department variants.
RRF_K = 60


def _rrf_contributions(points: list) -> dict[str, float]:
    """chunk_id -> this ranked list's RRF contribution."""
    return {p.payload["chunk_id"]: 1.0 / (RRF_K + rank) for rank, p in enumerate(points, start=1)}


def merge(variant_results: dict[tuple[str, str], list], limit: int) -> list:
    """Fuse every (sub_query, dept_id) result list by RRF score, dedupe by
    chunk_id, then apply a per-department quota: a department with any match
    keeps at least one slot in the merged output, so its score distribution
    can't be fully starved by a department with more/stronger matches.
    """
    scores: dict[str, float] = {}
    points_by_chunk: dict[str, object] = {}
    dept_by_chunk: dict[str, str] = {}

    for (_, dept_id), points in variant_results.items():
        for chunk_id, contribution in _rrf_contributions(points).items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
        for p in points:
            chunk_id = p.payload["chunk_id"]
            points_by_chunk.setdefault(chunk_id, p)
            dept_by_chunk[chunk_id] = dept_id

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked_ids = [cid for cid, _ in ranked]
    depts_with_matches = {dept_by_chunk[cid] for cid in ranked_ids}

    guaranteed: list[str] = []
    seen_depts: set[str] = set()
    for chunk_id in ranked_ids:
        dept_id = dept_by_chunk[chunk_id]
        if dept_id not in seen_depts:
            guaranteed.append(chunk_id)
            seen_depts.add(dept_id)
        if len(seen_depts) == len(depts_with_matches):
            break

    guaranteed_set = set(guaranteed)
    ordered_ids = guaranteed + [cid for cid in ranked_ids if cid not in guaranteed_set]
    return [points_by_chunk[cid] for cid in ordered_ids[:limit]]


if __name__ == "__main__":
    class _P:
        def __init__(self, chunk_id):
            self.payload = {"chunk_id": chunk_id}

    variants = {
        ("q", "dept-a"): [_P("a1"), _P("a2")],
        ("q", "dept-b"): [_P("b1")],
    }
    merged = merge(variants, limit=10)
    merged_ids = [p.payload["chunk_id"] for p in merged]
    assert set(merged_ids) == {"a1", "a2", "b1"}
    assert "b1" in merged_ids  # quota: dept-b's only match survives even though dept-a has 2 hits
    print("retrieval/merge.py: self-check passed")
