# retrieval/recency.py — §5.4: post-rerank score decay by indexed_at
import math
from datetime import datetime, timezone

from core.config import RECENCY_HALF_LIFE_DAYS

_LN2 = math.log(2)


def _age_days(indexed_at: str) -> float:
    ts = datetime.fromisoformat(indexed_at)
    now = datetime.now(timezone.utc)
    return max((now - ts).total_seconds() / 86400, 0.0)


def decay(rerank_score: float, indexed_at: str) -> float:
    age = _age_days(indexed_at)
    return rerank_score * math.exp(-_LN2 * age / RECENCY_HALF_LIFE_DAYS)


def apply_recency(scored_points: list[tuple[object, float]]) -> list[tuple[object, float]]:
    """[(point, rerank_score)] -> same, score decayed by chunk age, re-sorted."""
    adjusted = [(p, decay(score, p.payload["indexed_at"])) for p, score in scored_points]
    return sorted(adjusted, key=lambda ps: ps[1], reverse=True)


if __name__ == "__main__":
    now = datetime.now(timezone.utc).isoformat()
    assert decay(1.0, now) > 0.999  # ~zero age -> ~no decay
    old = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    assert decay(1.0, old) < 0.01  # decades old -> heavily decayed
    print("retrieval/recency.py: self-check passed")
