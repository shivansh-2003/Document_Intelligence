# image_pipeline.py
import json
import logging

from pydantic import BaseModel, ValidationError

from services.llm_service import generate
from utils.parsing_utils import RawElement

logger = logging.getLogger(__name__)

# image_utils.py — prompt only
IMAGE_PROMPT = """Describe this image for retrieval in a RAG system.
State the image type (diagram, chart, photo, technical drawing, graph, etc).
Then give a dense factual description: what it shows, any text/labels visible,
and any data values if it's a chart or graph. No preamble, no "the image shows"."""


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