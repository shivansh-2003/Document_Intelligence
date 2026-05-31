"""
image_pipeline.py
VLM-powered image summarization and metadata tagging.

Model: qwen3.6 (single multimodal model — vision + text)

The parsers already hand us each image (path / base64 / OCR text), so the
pipeline is just two steps per image:

  1. summarize_image()  → one strong prompt → concise, retrieval-ready summary
  2. build_metadata()   → assemble the metadata dict (incl. a cheap type guess)

Falls back to the OCR text when image bytes are unavailable.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ImageChunk:
    chunk_id:   str
    text:       str        # VLM summary — the embeddable content
    image_type: str        # chart | diagram | table | photo | equation | general
    ocr_text:   str        # raw text extracted by OCR / unstructured
    metadata:   dict[str, Any] = field(default_factory=dict)


IMAGE_TYPES   = {"chart", "diagram", "table", "photo", "equation", "general"}
DEFAULT_MODEL = "qwen3.6"   # single model for both vision and text tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    return "img_" + uuid.uuid4().hex[:8]


def _recency_score(upload_date: str | None) -> float:
    if not upload_date:
        return 1.0
    try:
        d = datetime.fromisoformat(str(upload_date)).date() if not isinstance(
            upload_date, (date, datetime)) else upload_date
        days = (date.today() - (d if isinstance(d, date) else d.date())).days
        return round(math.exp(-days / 180), 4)
    except Exception:
        return 1.0


def _load_pil_image(img_dict: dict) -> Any | None:
    try:
        from PIL import Image as PILImage
    except ImportError:
        return None
    meta = img_dict.get("metadata", {})
    image_path = meta.get("image_path") or meta.get("filename")
    if image_path:
        try:
            return PILImage.open(image_path).convert("RGB")
        except Exception:
            pass
    b64 = meta.get("image_base64") or meta.get("image_data")
    if b64:
        try:
            return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            pass
    return None


def _pil_to_base64(image: Any) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _ollama_chat(prompt: str, model: str, image: Any | None, max_tokens: int) -> str:
    """Single Ollama call. Sends the image when present, else text-only."""
    msg: dict[str, Any] = {"role": "user", "content": prompt}
    if image is not None:
        msg["images"] = [_pil_to_base64(image)]
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[msg],
            options={"num_predict": max_tokens, "temperature": 0.1},
        )
        return (response.get("message", {}).get("content", "") if isinstance(response, dict)
                else response.message.content).strip()
    except Exception as exc:
        logger.warning("Ollama call failed (model=%r, vision=%s): %s",
                       model, image is not None, exc)
        return ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = """\
You are describing an image so it can be retrieved later by search.

Write a concise, factual summary (3-5 sentences) that captures:
- What the image is (chart, diagram, table, photo, equation, screenshot, etc.)
- Its main subject or purpose
- The key data, labels, values, or text actually visible in it

Transcribe important numbers and labels exactly. Do not invent anything that is
not in the image."""

_TEXT_SUMMARY_PROMPT = """\
The following text was extracted from an image. Write a concise, factual
summary (3-5 sentences) describing what the image most likely shows and its key
content. Transcribe important numbers and labels exactly. Do not invent anything.

Extracted text:
{text}"""


# ---------------------------------------------------------------------------
# 1. Summarization
# ---------------------------------------------------------------------------

def summarize_image(image: Any | None, ocr_text: str, model: str) -> str:
    """Build a strong prompt and return a brief, retrieval-ready summary."""
    if image is not None:
        summary = _ollama_chat(_SUMMARY_PROMPT, model, image, max_tokens=512)
    elif ocr_text:
        summary = _ollama_chat(
            _TEXT_SUMMARY_PROMPT.format(text=ocr_text[:2000]),
            model, None, max_tokens=384,
        )
    else:
        summary = ""

    # Fall back to raw OCR text if the model returned nothing useful.
    if not summary and ocr_text:
        logger.info("Empty summary; falling back to OCR text")
        summary = ocr_text
    return summary


# ---------------------------------------------------------------------------
# 2. Metadata tagging
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS = {
    "chart":    ("chart", "graph", "plot", "histogram", "axis", "bar ", "pie "),
    "diagram":  ("diagram", "flowchart", "architecture", "uml", "topology", "workflow"),
    "table":    ("table", "column", "row", "spreadsheet"),
    "equation": ("equation", "formula", "latex", "theorem"),
    "photo":    ("photo", "photograph", "screenshot", "picture", "illustration"),
}


def _guess_type(text: str) -> str:
    """Cheap keyword heuristic — no extra LLM call."""
    low = text.lower()
    for img_type, words in _TYPE_KEYWORDS.items():
        if any(w in low for w in words):
            return img_type
    return "general"


def build_metadata(
    chunk_id:    str,
    summary:     str,
    ocr_text:    str,
    image_type:  str,
    img_index:   int,
    has_image:   bool,
    source_type: str,
    model:       str,
    img_meta:    dict,
    doc_meta:    dict,
) -> dict:
    """Assemble the metadata dict for an image chunk."""
    image_path = img_meta.get("image_path") or img_meta.get("filename", "")
    return {
        "chunk_id":        chunk_id,
        "doc_id":          doc_meta.get("doc_id", ""),
        "dept_id":         doc_meta.get("dept_id", ""),
        "doc_type":        source_type,
        "content_type":    "image",
        "chunk_strategy":  "single_image",
        "image_type":      image_type,
        "image_index":     img_index,
        "vlm_backend":     "ollama",
        "vlm_model":       model,
        "has_image_bytes": has_image,
        "description":     summary,
        "ocr_text":        ocr_text,
        "has_numbers":     bool(re.search(r'\d', summary + ocr_text)),
        "page_number":     img_meta.get("page_number"),
        "slide_number":    img_meta.get("slide_number") or img_meta.get("slide_index"),
        "image_path":      str(image_path) if image_path else None,
        "filename":        doc_meta.get("filename", ""),
        "upload_date":     str(doc_meta.get("upload_date", "")),
        "created_by":      doc_meta.get("created_by", ""),
        "recency_score":   _recency_score(doc_meta.get("upload_date")),
    }


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def _process_single_image(
    img_dict: dict,
    img_index: int,
    source_type: str,
    doc_meta: dict,
    model: str,
) -> ImageChunk:
    img_meta  = img_dict.get("metadata", {})
    ocr_text  = img_dict.get("text", "").strip()
    pil_image = _load_pil_image(img_dict)
    has_image = pil_image is not None

    logger.info("Processing image %d: has_image_bytes=%s ocr_text_len=%d",
                img_index, has_image, len(ocr_text))
    if not has_image and not ocr_text:
        logger.warning("Image %d has no bytes and no OCR text", img_index)

    summary    = summarize_image(pil_image, ocr_text, model)
    image_type = _guess_type(summary)
    logger.info("Image %d summarized (%d chars), type=%r", img_index, len(summary), image_type)

    chunk_id = _make_id()
    metadata = build_metadata(
        chunk_id, summary, ocr_text, image_type, img_index,
        has_image, source_type, model, img_meta, doc_meta,
    )

    embed_text = summary
    if ocr_text and ocr_text not in summary:
        embed_text = f"{summary}\n\nExtracted text: {ocr_text}"

    return ImageChunk(
        chunk_id=chunk_id,
        text=embed_text,
        image_type=image_type,
        ocr_text=ocr_text,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_images(
    images: list[dict],
    source_type: str,
    doc_meta: dict,
    vlm_backend: str = "ollama",
    vlm_config: dict | None = None,
) -> list[ImageChunk]:
    """
    Summarize and tag images using qwen3.6 (vision + text in one model).

    Args:
        images:      List of image dicts from ParsedDocument.images.
        source_type: "pdf" | "docx" | "pptx" | etc.
        doc_meta:    {doc_id, dept_id, filename, upload_date, created_by}
        vlm_backend: "ollama" (only supported backend)
        vlm_config:  {"model": "qwen3.6"} — override model tag if needed.

    Returns:
        List of ImageChunk objects, one per image.
    """
    if not images:
        logger.info("process_images: no images to process")
        return []

    model = (vlm_config or {}).get("model", DEFAULT_MODEL)
    logger.info("process_images: processing %d image(s) (model=%r)", len(images), model)

    chunks: list[ImageChunk] = []
    for idx, img_dict in enumerate(images):
        if not isinstance(img_dict, dict):
            logger.warning("Image at index %d is not a dict, skipping", idx)
            continue
        chunks.append(_process_single_image(img_dict, idx, source_type, doc_meta, model))

    logger.info("process_images complete: %d image chunks produced", len(chunks))
    return chunks
