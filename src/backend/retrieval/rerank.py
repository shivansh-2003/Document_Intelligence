# retrieval/rerank.py — §5.2: single rerank pass vs resolved_query
import asyncio
import logging

from fastembed.rerank.cross_encoder import TextCrossEncoder

from core.config import RERANK_MODEL, RERANK_TOP_K

logger = logging.getLogger(__name__)

_reranker: TextCrossEncoder | None = None


def _get_reranker() -> TextCrossEncoder:
    # Lazy singleton -- same reasoning as embedding_service._get_dense/_get_sparse:
    # importing this module must not trigger a model download.
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker %s", RERANK_MODEL)
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def _rerank_sync(query: str, points: list) -> list[tuple[object, float]]:
    texts = [p.payload["text"] for p in points]
    scores = list(_get_reranker().rerank(query, texts))
    return sorted(zip(points, scores), key=lambda ps: ps[1], reverse=True)[:RERANK_TOP_K]


async def rerank(resolved_query: str, points: list) -> list[tuple[object, float]]:
    """Single pass against resolved_query -- not one pass per sub_query. CPU-bound
    ONNX inference, same run_in_executor reasoning already documented in
    vector_store.upsert_chunks for the embedder: this must not block the event loop."""
    if not points:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _rerank_sync, resolved_query, points)
