# retrieval/confidence_gate.py — §6.1: threshold check, gates generation + cache writes
from core.config import CONFIDENCE_THRESHOLD


def passes(scored_points: list[tuple[object, float]]) -> bool:
    """Gates two independent things downstream (services/retrieval_service.py):
    whether generation.py runs at all, and whether tier 2/3 caches get written --
    a low-confidence result isn't cached and reused as if it were trustworthy."""
    if not scored_points:
        return False
    return scored_points[0][1] >= CONFIDENCE_THRESHOLD


if __name__ == "__main__":
    assert not passes([])
    assert passes([(None, 0.9)])
    assert not passes([(None, 0.1)])
    print("retrieval/confidence_gate.py: self-check passed")
