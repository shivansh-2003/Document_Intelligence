# image_pipeline.py
import json
import logging

from pydantic import BaseModel, ValidationError

from services.llm_service import generate
from utils.parsing_utils import RawElement

logger = logging.getLogger(__name__)

IMAGE_PROMPT = """You are describing an image for a RAG retrieval system. Someone will search for this
image using natural language — your description is the only thing that gets embedded and matched, the
original image stays available separately for visual lookup.

Step 1 — classify the image using this priority order (pick the first that applies):
  chart      — has axes, a legend, plotted data points, or bars/lines/slices representing values
  diagram    — shows components/nodes and how they connect (flowchart, architecture, process, org chart)
  table      — a grid of rows/columns rendered as an image rather than as an HTML table
  equation   — primarily mathematical notation
  photo      — a real-world photograph (people, objects, scenes, screenshots of physical things)
  general    — doesn't fit the above (logos, decorative graphics, icons, mixed collages)

Step 2 — write a dense description using the template for that type. Plain prose, no preamble
("This image shows..."), no markdown, no bullet points.

chart:
  - Chart type (bar / line / pie / scatter / area / etc).
  - Both axis labels and their units, exactly as printed.
  - Every legend entry / series name.
  - Named data points: for each series, call out the highest, lowest, first, and last values with
    their axis labels attached (e.g. "APAC revenue peaked at $42M in Q3 2024, falling to $31M by Q1
    2025" — not "revenue fluctuated"). Do not compute, round, or estimate a value that is not visibly
    labeled or clearly readable from gridlines — if a precise value isn't legible, say so rather than
    guessing.
  - Overall trend or comparison, only if the data actually supports it.

diagram:
  - Every node/component visible, using its exact on-image label.
  - Every connection between nodes, stated as "X -> Y" with the connection's label if one is drawn
    (arrow text, condition, protocol name).
  - Overall flow direction or entry/exit points if the diagram implies one.
  - Do not infer a relationship that isn't drawn as a line/arrow, even if it seems logically implied.

table (image-rendered):
  - Row and column count, and what each column represents.
  - Every visible header, resolved through merged/multi-level headers to its full path.
  - Standout values with row/column labels attached, same rule as chart — no invented figures.

equation:
  - Full transcription in LaTeX.
  - Plain-language meaning of each variable/symbol, only if inferable from surrounding labels — don't
    invent a domain interpretation the image doesn't support.

photo:
  - Every distinct object and its approximate position (left/right/foreground/background).
  - Setting/environment.
  - Any legible text, signage, or UI elements visible in the scene (summarized here; full verbatim
    transcription goes in text_in_image).

general:
  - Exhaustive factual description of every visible element. State plainly if the image has no
    retrievable informational content (e.g. a decorative divider or logo mark).

Rules that apply to every type:
- Every number, label, and word you cite must appear in the image exactly as shown — same casing,
  same units, same precision. No fabrication, no rounding, no "approximately" unless the image itself
  shows an approximation.
- 8 sentences max regardless of image complexity — for a dense image, prioritize the most notable/
  extreme elements over exhaustive coverage.
- text_in_image is separate from description: put every piece of visible text there verbatim, in
  reading order, even if you already summarized it in description. If there is no visible text, return
  an empty string — do not write "no text visible" in this field.
"""


class ImageDescription(BaseModel):
    type: str            # chart | diagram | table | equation | photo | general
    description: str     # leads with the type noun for keyword matchability
    text_in_image: str    # verbatim transcription of visible text, "" if none


def describe_image(el: RawElement) -> RawElement:
    if el.type not in ("Image", "Figure") or not el.metadata.image_base64:
        return el
    logger.info("Describing image element_id=%s type=%s", el.element_id, el.type)
    raw = generate(IMAGE_PROMPT, image_b64=el.metadata.image_base64, schema=ImageDescription)
    try:
        el.text = ImageDescription.model_validate_json(raw).model_dump_json()   # matches unstructured: text holds a JSON string
    except ValidationError:
        logger.warning("Malformed structured response for image element_id=%s; wrapping as unknown", el.element_id)
        el.text = json.dumps({"type": "unknown", "description": raw, "text_in_image": ""})   # ponytail: swallow malformed output, don't crash the batch
    return el


def describe_images(elements: list[RawElement]) -> list[RawElement]:
    images = [e for e in elements if e.type in ("Image", "Figure") and e.metadata.image_base64]
    logger.info("Describing %d image(s) of %d elements", len(images), len(elements))
    return [describe_image(e) for e in elements]