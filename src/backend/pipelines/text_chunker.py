"""
text_chunker.py
Adaptive text chunking pipeline for multi-source RAG documents.

Routing logic:
  pptx        → slide-level     (1 chunk per slide text block)
  audio/video → sentence window (1 sentence embedded + 3-sentence context)
  ≥2 markdown headers detected  → header-aware split (MarkdownHeaderTextSplitter
                                   → recursive sub-split for long sections)
  everything else               → recursive character split (512 tok / 64 overlap)

All metadata is extracted without LLM calls (pure heuristics + regex).
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — only fail if a strategy actually needs the missing library
# ---------------------------------------------------------------------------

def _tiktoken_encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def _langdetect(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text[:2000])  # first 2000 chars is enough
    except Exception:
        return "en"


def _nltk_sentences(text: str) -> list[str]:
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except ImportError:
        # Fallback: split on ". " boundaries
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _markdown_splitter():
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        return MarkdownHeaderTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import MarkdownHeaderTextSplitter
            return MarkdownHeaderTextSplitter
        except ImportError:
            logger.warning("MarkdownHeaderTextSplitter not available; will fall back to recursive splitter")
            return None


def _recursive_splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter
        except ImportError:
            logger.warning("RecursiveCharacterTextSplitter not available; will use naive word-split fallback")
            return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    chunk_id: str
    text: str                   # content to embed
    context_text: str           # sentence-window extended context (else same as text)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODER = None

def _count_tokens(text: str) -> int:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = _tiktoken_encoder()
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Rough fallback: ~4 chars per token
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Semantic density
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "these", "those", "it", "its",
})

def _semantic_density(text: str) -> float:
    tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if not tokens:
        return 0.0
    meaningful = [t for t in tokens if t not in _STOPWORDS]
    unique_meaningful = len(set(meaningful))
    return round(unique_meaningful / len(tokens), 3)


# ---------------------------------------------------------------------------
# Recency score
# ---------------------------------------------------------------------------

def _recency_score(upload_date: str | None) -> float:
    if not upload_date:
        return 1.0
    try:
        if isinstance(upload_date, (date, datetime)):
            d = upload_date
        else:
            d = datetime.fromisoformat(str(upload_date)).date()
        days = (date.today() - (d if isinstance(d, date) else d.date())).days
        return round(math.exp(-days / 180), 4)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

def _detect_strategy(text: str, source_type: str) -> str:
    if source_type == "pptx":
        return "slide_level"
    if source_type in ("audio", "video", "video_transcript"):
        return "sentence_window"
    header_count = len(re.findall(r'^#{1,3}\s+\S', text, re.MULTILINE))
    if header_count >= 2:
        return "markdown_header"
    return "recursive"


# ---------------------------------------------------------------------------
# Metadata builder (shared across strategies)
# ---------------------------------------------------------------------------

def _base_meta(
    chunk_id: str,
    text: str,
    strategy: str,
    source_type: str,
    doc_meta: dict,
    headings_path: list[str] | None = None,
    slide_number: int | None = None,
    page_number: int | None = None,
) -> dict:
    headings_path = headings_path or []
    return {
        # identity
        "chunk_id": chunk_id,
        "doc_id": doc_meta.get("doc_id", ""),
        "dept_id": doc_meta.get("dept_id", ""),
        # structure
        "doc_type": source_type,
        "content_type": "text",
        "chunk_strategy": strategy,
        "headings_path": headings_path,
        "hierarchy_level": len(headings_path),
        "parent_section": headings_path[-1] if headings_path else "",
        # location
        "page_number": page_number,
        "slide_number": slide_number,
        # semantic (zero LLM cost)
        "language": _langdetect(text),
        "has_numbers": bool(re.search(r'\d', text)),
        "has_citations": bool(re.search(r'\[[\d]+\]|\([\d]{4}\)', text)),
        "semantic_density": _semantic_density(text),
        "token_count": _count_tokens(text),
        # admin
        "filename": doc_meta.get("filename", ""),
        "upload_date": str(doc_meta.get("upload_date", "")),
        "created_by": doc_meta.get("created_by", ""),
        "recency_score": _recency_score(doc_meta.get("upload_date")),
    }


def _make_id() -> str:
    return "chnk_" + uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _chunk_slide_level(texts: list[str], source_type: str, doc_meta: dict) -> list[TextChunk]:
    """One chunk per slide text block; sub-split only if > 400 tokens."""
    MAX_SLIDE_TOKENS = 400
    chunks: list[TextChunk] = []

    for slide_idx, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue

        slide_num = slide_idx + 1

        if _count_tokens(text) <= MAX_SLIDE_TOKENS:
            cid = _make_id()
            chunks.append(TextChunk(
                chunk_id=cid,
                text=text,
                context_text=text,
                metadata=_base_meta(cid, text, "slide_level", source_type, doc_meta,
                                    slide_number=slide_num),
            ))
        else:
            # Sub-split long slides with recursive splitter
            sub = _chunk_recursive([text], source_type, doc_meta, slide_number=slide_num)
            for c in sub:
                c.metadata["chunk_strategy"] = "slide_level"
                c.metadata["slide_number"] = slide_num
            chunks.extend(sub)

    return chunks


def _chunk_sentence_window(texts: list[str], source_type: str, doc_meta: dict,
                            window: int = 3) -> list[TextChunk]:
    """Each sentence is the retrieval unit; surrounding ±window sentences are context."""
    full_text = "\n".join(t.strip() for t in texts if t.strip())
    sentences = _nltk_sentences(full_text)
    chunks: list[TextChunk] = []

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
        lo = max(0, i - window)
        hi = min(len(sentences), i + window + 1)
        context = " ".join(sentences[lo:hi])

        cid = _make_id()
        chunks.append(TextChunk(
            chunk_id=cid,
            text=sent,
            context_text=context,
            metadata={
                **_base_meta(cid, sent, "sentence_window", source_type, doc_meta),
                "sentence_index": i,
                "context_window": f"{lo}-{hi}",
            },
        ))

    return chunks


def _chunk_markdown_header(texts: list[str], source_type: str, doc_meta: dict) -> list[TextChunk]:
    """Split on markdown headers first; recursively sub-split long sections."""
    full_text = "\n\n".join(t.strip() for t in texts if t.strip())

    MarkdownHeaderTextSplitter = _markdown_splitter()
    if MarkdownHeaderTextSplitter is None:
        # Fallback if langchain not installed
        return _chunk_recursive(texts, source_type, doc_meta)

    header_splits = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=header_splits,
        strip_headers=False,
    )
    sections = splitter.split_text(full_text)

    chunks: list[TextChunk] = []
    MAX_SECTION_TOKENS = 512
    OVERLAP_TOKENS = 64

    RecursiveCharacterTextSplitter = _recursive_splitter()

    for section in sections:
        text = section.page_content.strip()
        if not text:
            continue

        # Build heading path from metadata keys h1/h2/h3
        meta_headers = section.metadata  # {"h1": "Section", "h2": "Sub"}
        headings_path = [v for k, v in sorted(meta_headers.items()) if v]

        if _count_tokens(text) <= MAX_SECTION_TOKENS or RecursiveCharacterTextSplitter is None:
            cid = _make_id()
            chunks.append(TextChunk(
                chunk_id=cid,
                text=text,
                context_text=text,
                metadata=_base_meta(cid, text, "markdown_header", source_type, doc_meta,
                                    headings_path=headings_path),
            ))
        else:
            # Long section: sub-split recursively, preserve headings_path
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_SECTION_TOKENS,
                chunk_overlap=OVERLAP_TOKENS,
                length_function=_count_tokens,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            sub_texts = sub_splitter.split_text(text)
            for sub in sub_texts:
                sub = sub.strip()
                if not sub:
                    continue
                cid = _make_id()
                chunks.append(TextChunk(
                    chunk_id=cid,
                    text=sub,
                    context_text=sub,
                    metadata=_base_meta(cid, sub, "markdown_header", source_type, doc_meta,
                                        headings_path=headings_path),
                ))

    return chunks


def _chunk_recursive(texts: list[str], source_type: str, doc_meta: dict,
                     slide_number: int | None = None,
                     page_number: int | None = None) -> list[TextChunk]:
    """Standard recursive character split with sentence-boundary separators."""
    full_text = "\n\n".join(t.strip() for t in texts if t.strip())

    RecursiveCharacterTextSplitter = _recursive_splitter()
    if RecursiveCharacterTextSplitter is None:
        # Hard fallback: naive fixed-size split
        words = full_text.split()
        raw_chunks = [" ".join(words[i:i+400]) for i in range(0, len(words), 400)]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            length_function=_count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(full_text)

    chunks: list[TextChunk] = []
    for raw in raw_chunks:
        raw = raw.strip()
        if not raw:
            continue
        cid = _make_id()
        chunks.append(TextChunk(
            chunk_id=cid,
            text=raw,
            context_text=raw,
            metadata=_base_meta(cid, raw, "recursive", source_type, doc_meta,
                                slide_number=slide_number, page_number=page_number),
        ))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    texts: list[str],
    source_type: str,
    doc_meta: dict,
) -> list[TextChunk]:
    """
    Chunk a list of text strings from a parsed document into retrieval-ready TextChunks.

    Args:
        texts:       List of text strings from ParsedDocument.texts.
        source_type: File source type — "pdf" | "docx" | "txt" | "pptx" | "url" | "audio".
        doc_meta:    Document metadata dict with keys:
                       doc_id, dept_id, filename, upload_date, created_by.

    Returns:
        List of TextChunk objects, each with an embeddable `text`, optional
        extended `context_text`, and a full `metadata` dict.
    """
    if not texts:
        logger.info("chunk_text called with empty texts list; returning []")
        return []

    combined = "\n\n".join(t.strip() for t in texts if t.strip())
    if not combined:
        logger.info("chunk_text: all texts were blank after stripping; returning []")
        return []

    strategy = _detect_strategy(combined, source_type)
    logger.info("Text chunking strategy selected: %r for source_type=%r", strategy, source_type)

    if strategy == "slide_level":
        chunks = _chunk_slide_level(texts, source_type, doc_meta)
    elif strategy == "sentence_window":
        chunks = _chunk_sentence_window(texts, source_type, doc_meta)
    elif strategy == "markdown_header":
        chunks = _chunk_markdown_header(texts, source_type, doc_meta)
    else:
        chunks = _chunk_recursive(texts, source_type, doc_meta)

    logger.info("chunk_text produced %d chunks using strategy %r", len(chunks), strategy)
    return chunks
