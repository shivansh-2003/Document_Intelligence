# image_pipeline.py
import json
import logging

from pydantic import BaseModel, ValidationError

from services.llm_service import generate
from utils.parsing_utils import RawElement

logger = logging.getLogger(__name__)

IMAGE_PROMPT = """You are describing an image for a RAG retrieval system. Someone will search for this
image using natural language — your description is the only thing that gets embedded and matched.

Respond with ONLY a JSON object, no markdown fences, no preamble, matching exactly this shape:
{
  "type": "<diagram|photo|chart|graph|technical drawing|screenshot|flowchart|table_image|other>",
  "description": "<see rules below>",
  "text_in_image": "<verbatim text/labels/values visible in the image, or empty string if none>"
}

Rules for "description":
- Chart or graph: name the axes, the variables plotted, and the key values or trend (e.g. peak, minimum,
  crossover point) with numbers attached where readable. Do not fabricate values you can't actually read.
- Diagram or flowchart: describe the sequence/flow of steps or components in order, naming each labeled
  node or box.
- Photo or screenshot: describe what is concretely shown — objects, layout, visible UI elements — not
  a subjective or aesthetic read.
- One dense paragraph, 3-6 sentences. No hedging language ("it appears", "possibly") unless the image
  is genuinely illegible — in that case say so directly instead of guessing.

If the image is decorative (logo, divider, icon with no informational content), set "type" to "other"
and keep "description" to one short sentence saying so — don't pad it."""


class ImageDescription(BaseModel):
    type: str
    description: str


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