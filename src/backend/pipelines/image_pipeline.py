# image_pipeline.py
import json
import logging

from pydantic import BaseModel, ValidationError

from services.llm_service import generate
from utils.parsing_utils import RawElement

logger = logging.getLogger(__name__)

IMAGE_PROMPT = """You are describing an image for a RAG retrieval system. Someone will search for this
image using natural language — your description is the only thing that gets embedded and matched, the
image itself is not searchable.

Respond with ONLY a JSON object, no markdown fences, no preamble, matching exactly this shape:
{
  "type": "<diagram|photo|chart|graph|technical drawing|screenshot|flowchart|table_image|other>",
  "description": "<see rules below>",
  "text_in_image": "<verbatim text/labels/values visible in the image, or empty string if none>"
}

Choosing "type" — pick the single best match, checking in this order:
1. table_image — the image is a table (grid of rows/columns) rendered as a picture, not markup.
2. flowchart — connected boxes/nodes with arrows describing a sequence of steps or a decision process.
3. diagram — labeled components and their relationships, but not a step-by-step sequence (e.g. system
   architecture, org chart, anatomy).
4. chart / graph — data plotted against axes (bar, line, pie, scatter).
5. technical drawing — schematic, blueprint, CAD, wiring, or a dimensioned drawing.
6. screenshot — a captured UI, application window, or webpage.
7. photo — a real-world photograph.
8. other — decorative (logo, divider, icon) or none of the above fit.

Rules for "description" (one dense paragraph, 3-6 sentences; lead with the type noun for keyword
matchability — e.g. "A flowchart showing..." not throat-clearing like "This is an image of a flowchart."):
- table_image: treat exactly like a table — name the row/column headers and the standout values, with
  numbers and labels exactly as shown. Do not compute, round, or infer a value that isn't printed.
- flowchart / diagram: describe the sequence or structure in order, naming every labeled node/box and
  each connection between them — what leads to what, what depends on what.
- chart / graph: name the axes, the variables plotted, and the key values or trend (peak, minimum,
  crossover point) with numbers attached where readable.
- technical drawing: name the object/system depicted, its labeled parts, and any dimensions, tolerances,
  or callouts that are legible.
- screenshot / photo: describe what is concretely shown — objects, layout, visible UI elements — not a
  subjective or aesthetic read.
- Do not fabricate a value, label, or connection you can't actually read. If part of the image is
  illegible, say so directly for that part instead of guessing — don't hedge the whole description
  ("it appears", "possibly") over one unclear detail.

If "type" is "other" (decorative), keep "description" to one short sentence saying so — don't pad it.

"text_in_image": transcribe verbatim — same casing, punctuation, and line breaks as shown. Do not
paraphrase or summarize text that appears in the image; that's what this field is for."""


class ImageDescription(BaseModel):
    type: str
    description: str
    text_in_image: str = ""


def describe_image(el: RawElement) -> RawElement:
    if el.type not in ("Image", "Figure") or not el.metadata.image_base64:
        return el
    logger.info("Describing image element_id=%s type=%s", el.element_id, el.type)
    raw = generate(IMAGE_PROMPT, image_b64=el.metadata.image_base64, schema=ImageDescription)
    try:
        el.text = ImageDescription.model_validate_json(raw).model_dump_json()   # matches unstructured: text holds a JSON string
    except ValidationError:
        logger.warning("Malformed structured response for image element_id=%s; wrapping as unknown", el.element_id)
        el.text = json.dumps({"type": "unknown", "description": raw})   # ponytail: swallow malformed output, don't crash the batch
    return el


def describe_images(elements: list[RawElement]) -> list[RawElement]:
    images = [e for e in elements if e.type in ("Image", "Figure") and e.metadata.image_base64]
    logger.info("Describing %d image(s) of %d elements", len(images), len(elements))
    return [describe_image(e) for e in elements]