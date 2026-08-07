# services/retrieval_service.py — orchestrator mirroring parsing_service.py's role:
# wires the full query -> answer request path end to end. See context/retrieval.md §9.
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from caching import response_cache, retrieval_cache
from core.config import FANOUT_TOP_K
from ingestion_versioning.corpus_version import get_versions
from models import User
from retrieval.confidence_gate import passes
from retrieval.conversation_slots import load_context, save_slots
from retrieval.fanout import fanout
from retrieval.generation import Citation, citations_from, generate_answer
from retrieval.merge import merge
from retrieval.query_transform import plan_query
from retrieval.recency import apply_recency
from retrieval.rerank import rerank
from retrieval.scope_resolver import resolve_scope

NOT_CONFIDENT_MESSAGE = "I don't have enough information to answer that confidently."


class RetrievalResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confident: bool


async def answer_query(
    db: AsyncSession,
    user: User,
    query: str,
    conversation_id: str | None = None,
    dept_hint: list[str] | None = None,
) -> RetrievalResponse:
    context = await load_context(conversation_id)
    plan = plan_query(query, context)
    dept_ids = await resolve_scope(db, user, dept_hint or [])
    versions = await get_versions(db, dept_ids)

    scored = await retrieval_cache.get_cached(plan.resolved_query, dept_ids, versions)
    if scored is None:
        variant_results = await fanout(plan.sub_queries, dept_ids)
        merged = merge(variant_results, limit=FANOUT_TOP_K)
        scored = await rerank(plan.resolved_query, merged)
        scored = apply_recency(scored)

    if not passes(scored):
        # Raw citations still returned -- near-matches are worth showing even
        # when confidence_gate blocks generation, see context/retrieval.md §6.1.
        # Neither cache tier is written and slots aren't updated on this path --
        # an unreliable result shouldn't be reused as if it were trustworthy.
        return RetrievalResponse(answer=NOT_CONFIDENT_MESSAGE, citations=citations_from(scored), confident=False)

    await retrieval_cache.set_cached(plan.resolved_query, dept_ids, versions, scored)

    cached_response = await response_cache.get_cached(plan.resolved_query, dept_ids, versions)
    if cached_response:
        answer = cached_response["answer"]
        citations = [Citation(**c) for c in cached_response["citations"]]
    else:
        answer, citations = generate_answer(plan.resolved_query, scored)
        await response_cache.set_cached(
            plan.resolved_query, dept_ids, versions, answer, [c.model_dump() for c in citations]
        )

    await save_slots(conversation_id, plan)
    return RetrievalResponse(answer=answer, citations=citations, confident=True)
