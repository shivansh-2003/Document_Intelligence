# retrieval/query_transform.py — query text -> QueryPlan (classify + decompose + resolve)
# See context/retrieval.md §3.
import re

from pydantic import BaseModel

from services.llm_service import generate

# Below this word count, with no multi-part marker, the query is single-intent --
# skip the LLM call entirely rather than pay a round trip to confirm the obvious.
SKIP_ROUTER_MAX_WORDS = 12
_MULTI_PART = re.compile(r"\band\b|\bvs\b|\bversus\b", re.IGNORECASE)


class SubQuery(BaseModel):
    text: str
    dept_hint: list[str] = []   # department NAMES -- never trusted as access, see scope_resolver.py


class QueryPlan(BaseModel):
    resolved_query: str          # coreference-resolved against prior conversation context
    sub_queries: list[SubQuery]
    doc_type_hint: str | None = None
    period_hint: str | None = None


PROMPT_TEMPLATE = """You are the query-understanding stage of a RAG retrieval system. Given a user's
question and, optionally, prior conversation context, produce a single JSON object:

1. resolved_query -- the question rewritten so it stands alone: resolve pronouns and ellipsis
   ("what about last quarter") using the prior context below. If the question already stands
   alone, resolved_query is the question unchanged.
2. sub_queries -- one entry per independent question actually being asked. Most questions are one
   sub-query. Split only when the question genuinely asks for multiple distinct things (e.g.
   "compare X and Y" is two: one about X, one about Y). Each sub_query may carry a dept_hint --
   department names explicitly mentioned or clearly implied -- leave empty if none.
3. doc_type_hint -- a document type mentioned or implied (e.g. "contract", "invoice"), else null.
4. period_hint -- a time period mentioned or implied (e.g. "Q3 2024", "last year"), else null.

Prior conversation context (may be empty):
{context}

User question:
{query}"""


def build_prompt(query: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(query=query, context=context or "(none)")


def _looks_single_intent(query: str) -> bool:
    return len(query.split()) <= SKIP_ROUTER_MAX_WORDS and not _MULTI_PART.search(query)


def plan_query(query: str, context: str = "") -> QueryPlan:
    """query text (+ prior conversation_slots context) -> QueryPlan.

    Skip-router: a short, single-intent query with no prior context bypasses the
    LLM call -- the plan is just the query itself, unresolved and undecomposed,
    since there's nothing a structured call would meaningfully add.
    """
    if _looks_single_intent(query) and not context:
        return QueryPlan(resolved_query=query, sub_queries=[SubQuery(text=query)])

    raw = generate(build_prompt(query, context), schema=QueryPlan)
    return QueryPlan.model_validate_json(raw)


if __name__ == "__main__":
    assert _looks_single_intent("What is the refund policy?")
    assert not _looks_single_intent("Compare the Q3 and Q4 revenue and summarize the trend")
    plan = plan_query("What is the refund policy?")
    assert plan.resolved_query == "What is the refund policy?"
    assert plan.sub_queries == [SubQuery(text="What is the refund policy?")]
    print("retrieval/query_transform.py: self-check passed")
