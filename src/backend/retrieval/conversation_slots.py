# retrieval/conversation_slots.py — §3.4: semantic-only slot store.
# period/topic/doc_type, department NAMES only -- never dept_id/role. Backed by
# Valkey (already introduced for caching/) since this is ephemeral, TTL'd session
# state, not a durable record -- no Postgres table/migration for it.
import json

from core.config import CACHE_TTL_CONVERSATION
from core.valkey_client import get_client
from retrieval.query_transform import QueryPlan

_KEY_PREFIX = "slots:"


def _key(conversation_id: str) -> str:
    return f"{_KEY_PREFIX}{conversation_id}"


async def load_context(conversation_id: str | None) -> str:
    """conversation_id -> a short text blob for query_transform's prompt context.
    Empty string if there's no conversation_id or nothing stored yet -- that's what
    lets the skip-router still bypass the LLM call on a conversation's first turn."""
    if not conversation_id:
        return ""
    raw = await get_client().get(_key(conversation_id))
    if not raw:
        return ""
    slots = json.loads(raw)
    return "; ".join(f"{k}: {v}" for k, v in slots.items() if v)


async def save_slots(conversation_id: str | None, plan: QueryPlan) -> None:
    """Persists only period/topic/doc_type + department NAMES pulled from the
    resolved plan -- never dept_id or role. This is read back as a hint by
    query_transform on the next turn; scope_resolver.py re-validates any
    department name against live memberships regardless, every single call --
    the slot store is never itself a source of truth for access.
    """
    if not conversation_id:
        return
    dept_names = sorted({name for sq in plan.sub_queries for name in sq.dept_hint})
    slots = {
        "topic": plan.resolved_query,
        "doc_type": plan.doc_type_hint,
        "period": plan.period_hint,
        "departments": ", ".join(dept_names) if dept_names else None,
    }
    await get_client().set(_key(conversation_id), json.dumps(slots), ex=CACHE_TTL_CONVERSATION)
