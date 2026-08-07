# retrieval/generation.py — §6.2: single cited-answer LLM call
from pydantic import BaseModel

from services.llm_service import generate

PROMPT_TEMPLATE = """You are answering a question using only the excerpts below, retrieved from the
user's own document set. Cite every claim with the bracketed id of the excerpt it came from, e.g.
[chnk_a1b2c3d4e5f6]. If the excerpts don't contain enough information to answer, say so plainly
instead of guessing -- do not use outside knowledge.

Question:
{query}

Excerpts:
{excerpts}"""


class Citation(BaseModel):
    chunk_id: str
    filename: str | None = None
    page_number: int | None = None


def _format_excerpt(point) -> str:
    p = point.payload
    return f"[{p['chunk_id']}] ({p.get('filename') or 'unknown'}): {p['text']}"


def build_prompt(query: str, scored_points: list[tuple[object, float]]) -> str:
    excerpts = "\n\n".join(_format_excerpt(p) for p, _ in scored_points)
    return PROMPT_TEMPLATE.format(query=query, excerpts=excerpts)


def citations_from(scored_points: list[tuple[object, float]]) -> list[Citation]:
    """Reused by services/retrieval_service.py on the low-confidence path too --
    the near-matches are worth showing even when confidence_gate blocks generation."""
    return [
        Citation(
            chunk_id=p.payload["chunk_id"],
            filename=p.payload.get("filename"),
            page_number=p.payload.get("page_number"),
        )
        for p, _ in scored_points
    ]


def generate_answer(resolved_query: str, scored_points: list[tuple[object, float]]) -> tuple[str, list[Citation]]:
    answer = generate(build_prompt(resolved_query, scored_points))
    return answer, citations_from(scored_points)
