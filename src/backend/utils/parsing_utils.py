# utils/parsing_utils.py
from uuid import uuid4
from pydantic import BaseModel
from unstructured.documents.elements import Element


class ElementMetadata(BaseModel):
    filetype: str | None = None
    languages: list[str] | None = None
    page_number: int | None = None
    text_as_html: str | None = None
    image_base64: str | None = None
    image_mime_type: str | None = None
    filename: str | None = None
    data_source: dict = {}


class RawElement(BaseModel):
    idx: int                        # reading order — ours, not unstructured's, kept for the pipeline
    type: str                       # el.category: "Table", "Image", "NarrativeText", ...
    element_id: str
    text: str = ""
    metadata: ElementMetadata


def to_raw(elements: list[Element], path: str) -> list[RawElement]:
    """unstructured Elements -> ordered RawElements, unstructured-shaped."""
    from pathlib import Path
    fname = Path(path).name
    out = []
    for i, el in enumerate(elements):
        m = el.metadata
        out.append(RawElement(
            idx=i,
            type=el.category,
            element_id=getattr(el, "id", None) or uuid4().hex,
            text=el.text or "",
            metadata=ElementMetadata(
                filetype=getattr(m, "filetype", None),
                languages=getattr(m, "languages", None),
                page_number=getattr(m, "page_number", None),
                text_as_html=getattr(m, "text_as_html", None),
                image_base64=getattr(m, "image_base64", None),
                image_mime_type=getattr(m, "image_mime_type", None),
                filename=fname,
                data_source={},
            ),
        ))
    return out


def insight(els: list[RawElement]) -> "Counter[str]":
    from collections import Counter
    return Counter(e.type for e in els)