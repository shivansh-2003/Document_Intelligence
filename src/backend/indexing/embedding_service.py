# indexing/embedding_service.py — chunk -> dense + sparse vectors
#
# Two models, not one: BGE-M3 natively produces dense+sparse+ColBERT in a single
# forward pass, but that unified interface only ships in the Rust fastembed crate.
# The Python fastembed package doesn't carry BGE-M3 as an ONNX model at all (checked
# against fastembed 0.8.0's list_supported_models() -- not present even after
# upgrading, despite BGE-M3 being a documented fastembed dense model in older docs).
# BAAI/bge-large-en-v1.5 substitutes: same family, same 1024-dim (no schema change),
# stays lightweight ONNX. Tradeoff vs true BGE-M3: English-only, ~512-token limit per
# chunk instead of 8192 -- text chunks are capped at 2000 chars (text_pipeline.MAX_CHARS)
# which is usually under 512 tokens but not guaranteed for dense prose.
#
# Escape hatch if this measurably hurts retrieval quality: swap to
# FlagEmbedding.BGEM3FlagModel(...).encode(texts, return_dense=True, return_sparse=True)
# for true BGE-M3 dense+sparse in one pass. Heavier (real torch model, ~2.3GB+,
# no ONNX optimization) -- don't reach for it until SPLADE++ hybrid is measurably
# insufficient, not just for a model-name mismatch.
import json
import logging
import re

from fastembed import SparseTextEmbedding, TextEmbedding

from pipelines.text_pipeline import Chunk

_TAG_RE = re.compile(r"<[^>]+>")

logger = logging.getLogger(__name__)

DENSE_MODEL = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
DENSE_SIZE = 1024

_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None


def _get_dense() -> TextEmbedding:
    # Lazy singleton: importing this module must not trigger a multi-GB model
    # download when a process only needs the parsing path.
    global _dense_model
    if _dense_model is None:
        logger.info("Loading dense embedder %s", DENSE_MODEL)
        _dense_model = TextEmbedding(model_name=DENSE_MODEL)
    return _dense_model


def _get_sparse() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        logger.info("Loading sparse embedder %s", SPARSE_MODEL)
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def chunk_embed_text(chunk: Chunk) -> str:
    """What actually gets embedded per chunk kind -- never a raw structural payload.
    Table HTML and image-JSON braces/keys are noise in vector space; the retrievable
    content is the LLM-generated prose (table summary, image description)."""
    if chunk.kind == "image":
        try:
            return json.loads(chunk.text).get("description", chunk.text)
        except json.JSONDecodeError:
            return chunk.text
    return chunk.text


def _clean_html_text(html: str) -> str:
    """Table HTML -> whitespace-joined cell text, tags stripped."""
    return " ".join(_TAG_RE.sub(" ", html).split())


def chunk_sparse_text(chunk: Chunk) -> str:
    """What gets sparse-encoded per chunk kind -- table chunks use cleaned cell
    text instead of chunk_embed_text()'s LLM summary. SPLADE (sparse) is a
    term-matching model: raw cell values ("$4.2M", "Q3") match exact-value
    queries far better than summary prose does. Every other kind is identical
    to chunk_embed_text() -- see context/retrieval.md §7.3."""
    if chunk.kind == "table" and chunk.metadata.text_as_html:
        return _clean_html_text(chunk.metadata.text_as_html)
    return chunk_embed_text(chunk)


def embed_batch(dense_texts: list[str], sparse_texts: list[str] | None = None) -> list[dict]:
    """dense_texts (+ optional distinct sparse_texts, same length/order) ->
    [{"dense": [...], "sparse": {"indices": [...], "values": [...]}}, ...].
    sparse_texts defaults to dense_texts -- every caller but
    vector_store.upsert_chunks wants the same text for both, and passing
    nothing here keeps their behavior identical to before this param existed."""
    sparse_texts = sparse_texts or dense_texts
    dense_vecs = list(_get_dense().embed(dense_texts))
    sparse_vecs = list(_get_sparse().embed(sparse_texts))
    return [
        {
            "dense": dense.tolist(),
            "sparse": {"indices": sparse.indices.tolist(), "values": sparse.values.tolist()},
        }
        for dense, sparse in zip(dense_vecs, sparse_vecs)
    ]
