"""
image_pipeline.py
─────────────────
Vision-language image analysis for the RAG ingestion stage.

For every image extracted from a parsed document this module produces one
``ImageChunk`` whose ``text`` is a dense, retrieval-optimized *analysis* of the
image — not a caption and not a summary. The VLM is prompted as a domain expert
and is asked to apply its own knowledge (recognise the architectural pattern,
name the chart type, transcribe the equation, infer the implied trend, etc.) so
the embedded text carries everything a downstream query could match on.

Design contract
───────────────
• Vision-only. Each image is analysed from real pixels. There is no OCR-only
  / text-model fallback path: if an image cannot be loaded into pixels it is
  skipped (and logged), never degraded into a lower-quality analysis.
• Two passes per image:
      1. classify   → one of {chart, diagram, table, photo, equation, general}
      2. deep parse → a type-tailored expert prompt
  Classification is a ~5-token call, so the second, expensive pass always runs
  against the most relevant prompt.
• Backend: Ollama with a vision-capable model (default ``qwen2.5-vl``).

Input image dict shape (from ParsedDocument.images):
    {"metadata": {... image_path | image_base64 | page_number | slide_number ...},
     "text": "<optional OCR text>"}
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

import ollama
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ImageChunk:
    chunk_id: str
    text: str            # VLM analysis — the embeddable content
    image_type: str      # chart | diagram | table | photo | equation | general
    ocr_text: str        # raw OCR text carried from the parsing stage
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Type vocabularies
# ---------------------------------------------------------------------------

IMAGE_TYPES = {"chart", "diagram", "table", "photo", "equation", "general"}

CHART_TYPES = {
    "bar", "line", "pie", "scatter", "heatmap", "area",
    "histogram", "waterfall", "gantt", "radar", "bubble", "other",
}

DEFAULT_VISION_MODEL = "qwen3.6"

# Generation knobs
_CLASSIFY_TOKENS = 24      # room for the word even if the model adds a short preamble
_ANALYSIS_TOKENS = 3072    # dense architecture diagrams need headroom; tune down for speed
_TEMPERATURE = 0.1

# Specific types are tested before falling back to "general".
_SPECIFIC_TYPES = ("diagram", "chart", "table", "equation", "photo")

# Some reasoning models still wrap output in <think>…</think> even with think=False.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompts — expert-analyst framing, knowledge-augmented, retrieval-tuned
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
Classify this image into exactly ONE word from this set:
chart, diagram, table, photo, equation, general

Guidance:
- chart    → bar/line/pie/scatter/area/heatmap/histogram/waterfall/radar plots
- diagram  → flowchart, software/system architecture, network topology, UML,
             ER model, mind map, sequence diagram, BPMN, state machine
- table    → grid of rows and columns
- equation → mathematical or scientific formula
- photo    → photograph, UI screenshot, illustration, logo
- general  → anything that fits none of the above

Reply with the single word only. No punctuation, no explanation."""

_ANALYSIS_PROMPTS: dict[str, str] = {
    "diagram": """\
You are a principal systems architect. Analyse this diagram for a technical
search index. Do NOT merely describe shapes — apply your engineering knowledge
to interpret it.

Report, exhaustively and concretely:
1. Diagram class and, if it is an architecture/system diagram, the architectural
   PATTERN by name (e.g. microservices, event-driven, layered/n-tier,
   hub-and-spoke, publish/subscribe, CQRS, pipes-and-filters, client-server,
   service mesh, lambda/kappa, hexagonal).
2. Every node/component/service: its label and the role it plays. Where an icon
   or label implies a specific technology (Kafka, Redis, Postgres, S3, nginx,
   Kubernetes, a load balancer, a CDN, a message queue, a cache, an API
   gateway), name that technology and justify it in a few words.
3. Every connection: source → target, the DIRECTION of flow, and the
   protocol/medium if shown or strongly implied (HTTP/REST, gRPC, async queue,
   event stream, SQL, websocket, file/batch).
4. The end-to-end data or control flow as an ordered sequence of steps.
5. Decision points, branches, loops, retries, and failure/rollback paths.
6. Trust / network / tenancy boundaries, security controls, and any
   scaling or replication notes (replicas, sharding, partitions, regions).
7. Layers, groupings, swimlanes, or subsystems and what each one owns.
8. Any title, legend, colour coding, and cardinality/annotations on edges.
Be specific and complete — every component and every edge must appear.""",

    "chart": """\
You are a data analyst. Extract this chart for a retrieval index so that a later
question about any value or trend in it can be answered from your text alone.

Report:
1. Chart type (bar/line/pie/scatter/area/heatmap/histogram/waterfall/radar/etc.).
2. Title, subtitle, and any source/footnote text.
3. X-axis: label, unit, scale (linear/log), full range, and tick values.
4. Y-axis: label, unit, scale, full range, and tick values.
5. Every data series (legend entry) and, for each, the value at every readable
   point. Transcribe actual numbers — do not round away meaning.
6. The time period or categories covered.
7. The quantitative trend per series: direction, magnitude of change
   (absolute and %), inflection points, peaks, troughs, and crossovers.
8. Outliers, anomalies, highlighted points, callouts, and annotations.
9. The single most important finding the chart communicates, stated plainly.
Include every number and label you can read.""",

    "table": """\
You are a data extraction specialist. Transcribe and interpret this table for a
retrieval index.

Report:
1. Title/caption and the subject of the table.
2. All column headers, in order, with units.
3. All row headers / category labels.
4. The full grid of cell values — transcribe every readable value. If the table
   is large, transcribe it as markdown rows.
5. Total / subtotal / summary rows and how they relate to the data.
6. Highlighted, merged, or special cells and what they signify.
7. Row and column counts.
8. The key comparison or pattern the table makes visible.
Prioritise verbatim values over prose.""",

    "equation": """\
You are a domain mathematician. Transcribe and explain this equation for a
retrieval index.

Report:
1. The complete expression transcribed in LaTeX.
2. A plain-language reading of what it computes.
3. Every variable, symbol, subscript, superscript, and operator and its meaning.
4. Units of each quantity where determinable.
5. The field/domain it belongs to (physics, statistics, finance, ML, control,
   thermodynamics, etc.) and, if it is a named/standard equation, its NAME and
   what it is used for.
6. Any constraints, boundary conditions, or surrounding context text.""",

    "photo": """\
You are a forensic image analyst. Describe this image exhaustively for a
retrieval index.

Report:
1. What the image depicts overall and its purpose/context.
2. Every distinct object, person, UI element, or region present.
3. ALL visible text transcribed verbatim (labels, signs, captions, watermarks,
   UI strings, code, numbers).
4. Setting / environment / platform.
5. Any embedded chart, table, diagram, or data and its values.
6. Spatial layout and relationships between elements.
7. Colours and visual hierarchy where they carry meaning.
This text is the only way the image will be found — be thorough and specific.""",

    "general": """\
You are an expert visual analyst. Analyse this image exhaustively for a
retrieval index, applying domain knowledge to interpret — not just describe —
what you see.

Report:
1. What the image is and the concept or message it conveys.
2. Every visible element/component and its role.
3. ALL visible text transcribed verbatim, including numbers and measurements.
4. Any structured content (chart, table, diagram, equation) and its data.
5. Relationships, flow, or hierarchy among elements.
6. Colour coding, legends, and annotations and their meaning.
Include every detail a future query might search for.""",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    return "img_" + uuid.uuid4().hex[:8]


def _recency_score(upload_date: str | None) -> float:
    """exp(-days_since_upload / 180); 1.0 when the date is unknown."""
    if not upload_date:
        return 1.0
    try:
        if isinstance(upload_date, (date, datetime)):
            d = upload_date if isinstance(upload_date, date) else upload_date.date()
        else:
            d = datetime.fromisoformat(str(upload_date)).date()
        days = (date.today() - d).days
        return round(math.exp(-days / 180), 4)
    except (ValueError, TypeError):
        return 1.0


def _load_image(img_dict: dict) -> PILImage.Image | None:
    """
    Load real pixels for an image from ``image_path`` or ``image_base64``.
    Returns None when no usable bytes are present — the caller skips it.
    """
    meta = img_dict.get("metadata", {})

    path = meta.get("image_path") or meta.get("filename")
    if path:
        try:
            return PILImage.open(path).convert("RGB")
        except (OSError, ValueError) as exc:
            logger.debug("Could not open image path %s: %s", path, exc)

    b64 = meta.get("image_base64") or meta.get("image_data")
    if b64:
        try:
            return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except (OSError, ValueError, base64.binascii.Error) as exc:
            logger.debug("Could not decode base64 image: %s", exc)

    return None


def _to_base64_png(image: PILImage.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _vlm(image_b64: str, prompt: str, model: str, max_tokens: int) -> str:
    """
    Single Ollama vision call.

    ``think=False`` disables chain-of-thought on reasoning-capable models
    (the Qwen3 family, etc.). For a structured extraction task we never want
    thinking tokens: on the classification pass they push the answer word out
    of reach, and on the analysis pass they silently consume the num_predict
    budget — which is exactly what truncates long descriptions. Requires a
    recent Ollama; the regex strip covers models that still emit the tags.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
        options={"num_predict": max_tokens, "temperature": _TEMPERATURE},
        think=False,
    )
    return _THINK_RE.sub("", response.message.content or "").strip()


def _classify(image_b64: str, model: str) -> str:
    """
    Route the image to a type by scanning for a type keyword anywhere in the
    reply — robust to a verbose or preamble-prefixed answer. Specific types win
    over the ``general`` catch-all.
    """
    raw = _vlm(image_b64, _CLASSIFY_PROMPT, model, _CLASSIFY_TOKENS).lower()
    for t in _SPECIFIC_TYPES:
        if re.search(rf"\b{t}s?\b", raw):
            return t
    return "general"


# ---------------------------------------------------------------------------
# Structured metadata extraction (knowledge mined from the VLM analysis)
# ---------------------------------------------------------------------------

def _extract_trend(text: str) -> str | None:
    low = text.lower()
    if any(w in low for w in ("increasing", "upward", "rising", "growing", "uptrend")):
        return "increasing"
    if any(w in low for w in ("decreasing", "downward", "falling", "declining", "downtrend")):
        return "decreasing"
    if any(w in low for w in ("stable", "flat", "constant", "unchanged", "plateau")):
        return "stable"
    return None


def _extract_chart_meta(description: str) -> dict[str, Any]:
    low = description.lower()
    chart_type = next((ct for ct in CHART_TYPES if ct in low), "other")

    axis_labels: dict[str, str] = {}
    for axis in ("x", "y"):
        m = re.search(rf"{axis}[-\s]?axis[:\s]+([^\n,\.]+)", description, re.IGNORECASE)
        if m:
            axis_labels[axis] = m.group(1).strip()

    key_values = re.findall(
        r"[\$£€]?\d[\d,\.]*\s*(?:M|B|K|%|million|billion|thousand|bn|mn)?",
        description,
    )[:12]

    return {
        "chart_type": chart_type,
        "axis_labels": axis_labels or None,
        "key_values": key_values or None,
        "trend": _extract_trend(description),
    }


def _extract_diagram_meta(description: str) -> dict[str, Any]:
    # Capitalised multi-word phrases tend to be named components/services.
    candidates = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b", description)
    seen: set[str] = set()
    entities: list[str] = []
    for c in candidates:
        key = c.lower()
        if key not in seen and len(c) > 2:
            seen.add(key)
            entities.append(c)

    # Surface a named architectural pattern if the VLM identified one.
    pattern = None
    for p in ("microservices", "event-driven", "publish/subscribe", "pub/sub",
              "layered", "n-tier", "hub-and-spoke", "service mesh", "cqrs",
              "client-server", "pipes-and-filters", "hexagonal", "serverless",
              "monolith", "lambda architecture", "kappa architecture"):
        if p in description.lower():
            pattern = p
            break

    return {
        "entities_mentioned": entities[:25] or None,
        "architecture_pattern": pattern,
    }


def _type_specific_meta(image_type: str, description: str) -> dict[str, Any]:
    if image_type == "chart":
        return _extract_chart_meta(description)
    if image_type == "diagram":
        return _extract_diagram_meta(description)
    return {}


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def _process_single_image(
    img_dict: dict,
    img_index: int,
    source_type: str,
    doc_meta: dict,
    vision_model: str,
) -> ImageChunk | None:
    """Analyse one image. Returns None when the image has no loadable pixels."""
    image = _load_image(img_dict)
    if image is None:
        logger.warning("Image %d: no loadable pixels (image_path/image_base64); skipping.",
                       img_index)
        return None

    meta = img_dict.get("metadata", {})
    ocr_text = (img_dict.get("text") or "").strip()
    image_b64 = _to_base64_png(image)

    # Pass 1 — classification (keyword-scan, robust to a verbose reply)
    image_type = _classify(image_b64, vision_model)
    logger.debug("Image %d classified as '%s'.", img_index, image_type)

    # Pass 2 — type-tailored deep analysis
    description = _vlm(
        image_b64, _ANALYSIS_PROMPTS[image_type], vision_model, _ANALYSIS_TOKENS
    )

    type_meta = _type_specific_meta(image_type, description)
    chunk_id = _make_id()

    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "doc_id": doc_meta.get("doc_id", ""),
        "dept_id": doc_meta.get("dept_id", ""),
        "doc_type": source_type,
        "content_type": "image",
        "chunk_strategy": "single_image",
        "image_type": image_type,
        "image_index": img_index,
        "vlm_backend": "ollama",
        "vlm_model": vision_model,
        "description": description,
        "ocr_text": ocr_text,
        "has_numbers": bool(re.search(r"\d", description + ocr_text)),
        "page_number": meta.get("page_number"),
        "slide_number": meta.get("slide_number") or meta.get("slide_index"),
        "image_path": str(meta.get("image_path") or meta.get("filename") or "") or None,
        "filename": doc_meta.get("filename", ""),
        "upload_date": str(doc_meta.get("upload_date", "")),
        "created_by": doc_meta.get("created_by", ""),
        "recency_score": _recency_score(doc_meta.get("upload_date")),
        **type_meta,
    }

    embed_text = description
    if ocr_text and ocr_text not in description:
        embed_text = f"{description}\n\nExtracted text: {ocr_text}"

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
    Analyse images from a parsed document with a vision-language model.

    Args:
        images:      Image dicts from ParsedDocument.images
                     ({"metadata": {...}, "text": str}).
        source_type: "pdf" | "docx" | "pptx" | ...
        doc_meta:    {doc_id, dept_id, filename, upload_date, created_by}.
        vlm_backend: Vision backend. Only "ollama" is supported.
        vlm_config:  {"vision_model": "qwen2.5-vl"}. Defaults applied if omitted.

    Returns:
        One ImageChunk per successfully analysed image. Images without loadable
        pixels are skipped (and logged), not degraded.
    """
    if not images:
        return []
    if vlm_backend != "ollama":
        raise ValueError(f"Unsupported vlm_backend {vlm_backend!r}; only 'ollama' is supported.")

    vision_model = (vlm_config or {}).get("vision_model", DEFAULT_VISION_MODEL)
    logger.info("Analysing %d image(s) with Ollama model '%s'.", len(images), vision_model)

    chunks: list[ImageChunk] = []
    skipped = 0
    for idx, img_dict in enumerate(images):
        if not isinstance(img_dict, dict):
            logger.debug("Image %d: not a dict (%s); skipping.", idx, type(img_dict).__name__)
            skipped += 1
            continue
        try:
            chunk = _process_single_image(img_dict, idx, source_type, doc_meta, vision_model)
        except Exception:
            logger.exception("Image %d: VLM analysis failed; skipping.", idx)
            skipped += 1
            continue
        if chunk is None:
            skipped += 1
            continue
        chunks.append(chunk)

    logger.info("Produced %d image chunk(s); %d skipped.", len(chunks), skipped)
    return chunks