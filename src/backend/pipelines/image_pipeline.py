# image_pipeline.py
import json
from services.llm_service import generate
from utils.parsing_utils import RawElement

# image_utils.py — prompt only
IMAGE_PROMPT = """Describe this image for retrieval in a RAG system.
State the image type (diagram, chart, photo, technical drawing, graph, etc).
Then give a dense factual description: what it shows, any text/labels visible,
and any data values if it's a chart or graph. No preamble, no "the image shows"."""


def describe_image(el: RawElement) -> RawElement:
    if el.type not in ("Image", "Figure") or not el.metadata.image_base64:
        return el
    raw = generate(IMAGE_PROMPT, image_b64=el.metadata.image_base64)
    try:
        parsed = json.loads(raw)                     # validate shape, don't trust the model blindly
        el.text = json.dumps(parsed)                  # matches unstructured: text holds a JSON string
    except json.JSONDecodeError:
        el.text = json.dumps({"type": "unknown", "description": raw})   # ponytail: swallow malformed output, don't crash the batch
    return el


def describe_images(elements: list[RawElement]) -> list[RawElement]:
    return [describe_image(e) for e in elements]