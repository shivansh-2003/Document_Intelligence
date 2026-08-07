# core/valkey_client.py — single Valkey client, lazy singleton.
# Backs caching/ (3 tiers, context/retrieval.md §8) and retrieval/conversation_slots.py.
from valkey.asyncio import Valkey

from core.config import VALKEY_URL

_client: Valkey | None = None


def get_client() -> Valkey:
    """Lazy singleton -- same reasoning as vectorstore/vector_store.py's module-level
    _client: importing a module that uses this must not require Valkey to already be
    reachable."""
    global _client
    if _client is None:
        _client = Valkey.from_url(VALKEY_URL, decode_responses=True)
    return _client
