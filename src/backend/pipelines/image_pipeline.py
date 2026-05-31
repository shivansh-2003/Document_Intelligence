"""
image_pipeline.py
VLM-powered image analysis and metadata tagging pipeline.

For each image from ParsedDocument.images:
  1. Classify image type: chart | diagram | table | photo | equation | general
  2. Run two-pass VLM prompting: classification → type-tailored deep analysis
  3. Extract structured metadata by image type
     - charts:   axis labels, key values, chart type, trend direction
     - diagrams: entities mentioned, relationships, process steps
     - general:  full exhaustive description + all visible text

Default VLM backend: Ollama with qwen2.5-vl (vision-capable Qwen model).
For image-type classification, qwen3 (text only) is used on the OCR/text field
when no image bytes are available.

Input images from ParsedDocument.images format:
  {"metadata": dict, "text": str}
  where metadata may contain:
    image_path      → path to extracted image file
    image_base64    → base64-encoded image bytes
    page_number     → source page
    slide_number    → source slide (for PPTX)
    coordinates     → bounding box dict
"""

from __future__ import annotations

import base64
import io
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ImageChunk:
    chunk_id: str
    text: str           # VLM description — the embeddable content
    image_type: str     # chart | diagram | table | photo | equation | general
    ocr_text: str       # raw text extracted by OCR or unstructured
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Image type enum values
# ---------------------------------------------------------------------------

IMAGE_TYPES = {"chart", "diagram", "table", "photo", "equation", "general"}

# Chart sub-types
CHART_TYPES = {"bar", "line", "pie", "scatter", "heatmap", "area",
               "histogram", "waterfall", "gantt", "radar", "other"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    return "img_" + uuid.uuid4().hex[:8]


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


def _load_pil_image(img_dict: dict) -> Any | None:
    """
    Try to load a PIL Image from the image dict.
    Checks: metadata.image_path, metadata.image_base64, metadata.image_data
    Returns None if no image bytes are available.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        return None

    meta = img_dict.get("metadata", {})

    # Try image_path first
    image_path = meta.get("image_path") or meta.get("filename")
    if image_path:
        try:
            return PILImage.open(image_path).convert("RGB")
        except Exception:
            pass

    # Try base64
    b64 = meta.get("image_base64") or meta.get("image_data")
    if b64:
        try:
            img_bytes = base64.b64decode(b64)
            return PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            pass

    return None


def _pil_to_base64(image: Any) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ---------------------------------------------------------------------------
# VLM backend (Ollama, minimal inline — no dep on backend/services/)
# ---------------------------------------------------------------------------

def _ollama_describe(image: Any, prompt: str, model: str, max_tokens: int = 1024) -> str:
    """Call Ollama vision model. image is a PIL Image."""
    try:
        import ollama
        img_b64 = _pil_to_base64(image)
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }],
            options={"num_predict": max_tokens, "temperature": 0.1},
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "").strip()
        return response.message.content.strip()
    except Exception:
        return ""


def _ollama_text(prompt: str, model: str, max_tokens: int = 512) -> str:
    """Call Ollama text model (no image). Used when no image bytes available."""
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": max_tokens, "temperature": 0.1},
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "").strip()
        return response.message.content.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT_VISION = """\
Look at this image and classify it into exactly one of these types:
chart, diagram, table, photo, equation, general

Respond with a single word only. No explanation.
- chart: bar chart, line graph, pie chart, scatter plot, heatmap, histogram
- diagram: flowchart, architecture diagram, UML, mind map, network topology
- table: tabular data with rows and columns
- photo: real-world photograph, screenshot of UI, illustration
- equation: mathematical formula or expression
- general: any image that doesn't fit the above
"""

_CLASSIFY_PROMPT_TEXT = """\
Based on this text description of an image, classify it into exactly one type:
chart, diagram, table, photo, equation, general

Text: {text}

Respond with a single word only. No explanation.
"""

_DEEP_PROMPTS: dict[str, str] = {
    "chart": """\
Analyze this chart exhaustively for retrieval. Include:
1. Chart type (bar/line/pie/scatter/etc.)
2. Title (if visible)
3. X-axis label and values/range
4. Y-axis label and values/range
5. All data series names (legend items)
6. Key data points and values visible
7. Overall trend or pattern (increasing/decreasing/stable/cyclical)
8. Any annotations, callouts, or highlighted values
9. Time period covered (if applicable)
10. Source or footnotes (if visible)
Be exhaustive — include every number and label you can read.""",

    "diagram": """\
Analyze this diagram exhaustively for retrieval. Include:
1. Type of diagram (flowchart/architecture/UML/mind map/network/etc.)
2. Title (if visible)
3. All labeled nodes, components, or entities
4. All relationships, arrows, or connections between components
5. Process flow direction (if it's a flowchart)
6. Decision points or branches
7. Any text labels on connections
8. Color coding and what each color means (if apparent)
9. Overall system or process being depicted
Be exhaustive — list every component and relationship you can identify.""",

    "table": """\
Describe this table exhaustively for retrieval. Include:
1. Title or heading (if visible)
2. Column headers (all of them)
3. Row headers or categories (if present)
4. All visible data values, especially key numbers
5. Any totals, subtotals, or summary rows
6. Units of measurement
7. Any highlighted or special cells
8. Number of rows and columns
Transcribe as much of the actual data as you can read.""",

    "photo": """\
Describe this image exhaustively for retrieval. Include:
1. What the image shows overall
2. All visible objects, people, or elements
3. Any visible text (signs, labels, captions, watermarks)
4. Setting or environment
5. Any data, numbers, or statistics visible
6. Colors and visual layout
7. Any charts, graphs, or diagrams embedded in the image
Be thorough — this description is the only way this image will be found.""",

    "equation": """\
Describe this equation or formula exhaustively for retrieval. Include:
1. The full equation transcribed in text (use LaTeX if possible)
2. What each variable or symbol represents (if labeled)
3. The field or domain this equation belongs to (physics, statistics, finance, etc.)
4. Any surrounding context text visible
5. Units of variables (if shown)
6. Any subscripts, superscripts, or special notation""",

    "general": """\
Describe this image exhaustively for retrieval. Include:
1. What the image shows overall
2. All visible objects, elements, or components
3. All visible text (transcribe it exactly)
4. Any data, numbers, statistics, or measurements visible
5. Color coding or visual hierarchy
6. Spatial relationships between elements
7. Any charts, tables, or structured data present
Be thorough and specific — include every detail that could be searched.""",
}


# ---------------------------------------------------------------------------
# Structured metadata extraction from VLM response
# ---------------------------------------------------------------------------

def _extract_chart_meta(description: str) -> dict:
    """Extract structured fields from a chart description."""
    chart_type = "other"
    for ct in CHART_TYPES:
        if ct in description.lower():
            chart_type = ct
            break

    # Extract axis labels (heuristic: look for "X-axis: ..." or "x axis:")
    axis_labels: dict[str, str] = {}
    for axis in ("x", "y"):
        m = re.search(rf'{axis}[- ]axis[:\s]+([^\n,\.]+)', description, re.IGNORECASE)
        if m:
            axis_labels[axis] = m.group(1).strip()

    # Extract key values: patterns like "$3.2M", "62%", "1,234"
    key_values = re.findall(
        r'(?:[\$£€]?\d[\d,\.]+\s*(?:M|B|K|%|million|billion|thousand)?)',
        description
    )[:10]  # cap at 10

    return {
        "chart_type": chart_type,
        "axis_labels": axis_labels or None,
        "key_values": key_values or None,
        "trend": _extract_trend(description),
    }


def _extract_trend(text: str) -> str | None:
    for word in ("increasing", "upward", "rising", "growing"):
        if word in text.lower():
            return "increasing"
    for word in ("decreasing", "downward", "falling", "declining"):
        if word in text.lower():
            return "decreasing"
    for word in ("stable", "flat", "constant", "unchanged"):
        if word in text.lower():
            return "stable"
    return None


def _extract_diagram_meta(description: str) -> dict:
    # Extract entity names: capitalized multi-word phrases often = components
    entities = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', description)
    # Deduplicate, keep only meaningful ones (len > 2)
    seen: set[str] = set()
    unique_entities: list[str] = []
    for e in entities:
        e_lower = e.lower()
        if e_lower not in seen and len(e) > 2:
            seen.add(e_lower)
            unique_entities.append(e)

    return {
        "entities_mentioned": unique_entities[:20] or None,  # cap at 20
    }


def _type_specific_meta(image_type: str, description: str) -> dict:
    if image_type == "chart":
        return _extract_chart_meta(description)
    if image_type == "diagram":
        return _extract_diagram_meta(description)
    return {}


# ---------------------------------------------------------------------------
# Core per-image processing
# ---------------------------------------------------------------------------

def _process_single_image(
    img_dict: dict,
    img_index: int,
    source_type: str,
    doc_meta: dict,
    vlm_model: str,
    text_model: str,
) -> ImageChunk:
    meta = img_dict.get("metadata", {})
    ocr_text = img_dict.get("text", "").strip()

    # Source location
    page_number = meta.get("page_number")
    slide_number = meta.get("slide_number") or meta.get("slide_index")
    image_path = meta.get("image_path") or meta.get("filename", "")

    # Try to load actual image
    pil_image = _load_pil_image(img_dict)
    has_image = pil_image is not None

    # --- Step 1: Classify image type ---
    if has_image:
        raw_type = _ollama_describe(pil_image, _CLASSIFY_PROMPT_VISION, vlm_model, max_tokens=10)
    elif ocr_text:
        raw_type = _ollama_text(
            _CLASSIFY_PROMPT_TEXT.format(text=ocr_text[:500]), text_model, max_tokens=10
        )
    else:
        raw_type = "general"

    raw_type = raw_type.strip().lower().split()[0] if raw_type.strip() else "general"
    image_type = raw_type if raw_type in IMAGE_TYPES else "general"

    # --- Step 2: Deep analysis ---
    deep_prompt = _DEEP_PROMPTS.get(image_type, _DEEP_PROMPTS["general"])

    if has_image:
        description = _ollama_describe(pil_image, deep_prompt, vlm_model, max_tokens=1024)
    elif ocr_text:
        # Text-only fallback: ask text model to expand on OCR content
        description = _ollama_text(
            f"Based on this text extracted from an image ({image_type}), "
            f"provide an exhaustive description for retrieval:\n\n{ocr_text[:2000]}",
            text_model, max_tokens=512
        )
    else:
        description = ocr_text  # nothing to work with

    if not description and ocr_text:
        description = ocr_text

    # --- Step 3: Structured metadata ---
    type_meta = _type_specific_meta(image_type, description)

    chunk_id = _make_id()
    metadata = {
        "chunk_id": chunk_id,
        "doc_id": doc_meta.get("doc_id", ""),
        "dept_id": doc_meta.get("dept_id", ""),
        "doc_type": source_type,
        "content_type": "image",
        "chunk_strategy": "single_image",
        "image_type": image_type,
        "image_index": img_index,
        "vlm_backend": "ollama",
        "vlm_model": vlm_model if has_image else text_model,
        "has_image_bytes": has_image,
        "description": description,
        "ocr_text": ocr_text,
        "has_numbers": bool(re.search(r'\d', description + ocr_text)),
        "page_number": page_number,
        "slide_number": slide_number,
        "image_path": str(image_path) if image_path else None,
        "filename": doc_meta.get("filename", ""),
        "upload_date": str(doc_meta.get("upload_date", "")),
        "created_by": doc_meta.get("created_by", ""),
        "recency_score": _recency_score(doc_meta.get("upload_date")),
        **type_meta,
    }

    # The embeddable text = VLM description + OCR (de-duplicate if they overlap)
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
    Analyze and tag images from a parsed document using a VLM.

    Args:
        images:      List of image dicts from ParsedDocument.images.
                     Each dict: {"metadata": {...}, "text": str}
                     metadata may include image_path, image_base64, page_number, etc.
        source_type: File source type — "pdf" | "docx" | "pptx" | etc.
        doc_meta:    Document metadata dict with keys:
                       doc_id, dept_id, filename, upload_date, created_by.
        vlm_backend: Vision model backend. Currently supported: "ollama".
        vlm_config:  Backend configuration dict. For ollama:
                       {"vision_model": "qwen2.5-vl", "text_model": "qwen3"}
                     Defaults: vision_model="qwen2.5-vl", text_model="qwen3"

    Returns:
        List of ImageChunk objects, one per image.
    """
    if not images:
        return []

    cfg = vlm_config or {}
    # qwen2.5-vl is the vision-capable Qwen model on Ollama
    # qwen3 is the text model for fallback classification when no image bytes
    vision_model = cfg.get("vision_model", "qwen2.5-vl")
    text_model = cfg.get("text_model", "qwen3")

    chunks: list[ImageChunk] = []
    for idx, img_dict in enumerate(images):
        if not isinstance(img_dict, dict):
            continue
        chunk = _process_single_image(
            img_dict, idx, source_type, doc_meta, vision_model, text_model
        )
        chunks.append(chunk)

    return chunks
