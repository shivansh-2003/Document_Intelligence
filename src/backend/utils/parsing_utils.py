# utils/parsing_utils.py
from collections import Counter
from uuid import uuid4

from pydantic import BaseModel
from unstructured.documents.elements import Element


class ElementMetadata(BaseModel):
    filetype: str | None = None
    languages: list[str] | None = None
    page_number: int | None = None          # None for txt/md/html — paginationless sources
    category_depth: int | None = None       # Title elements: heading level (0 = h1, 1 = h2, ...)
    text_as_html: str | None = None         # Table only
    image_base64: str | None = None         # Image only
    image_mime_type: str | None = None      # Image only
    filename: str | None = None             # stamped by parsing_service, not here


class RawElement(BaseModel):
    idx: int                        # reading order — ours, not unstructured's, kept for the pipeline
    type: str                       # el.category: "Table", "Image", "NarrativeText", ...
    element_id: str
    text: str = ""
    metadata: ElementMetadata


def to_raw(elements: list[Element]) -> list[RawElement]:
    """unstructured Elements -> ordered RawElements, unstructured-shaped.

    Source-agnostic on purpose. Every unstructured partitioner — pdf, docx, pptx,
    text, md, html — emits the same Element categories, so this one function serves
    all of them. It takes no path: filename is stamped by parsing_service, the only
    layer that knows whether the source is a file, a URL, or an S3 key. Fields absent
    for a given source (page_number on a .txt, image_base64 on html) come back None,
    and the downstream pipelines already guard on that.
    """
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
                category_depth=getattr(m, "category_depth", None),
                text_as_html=getattr(m, "text_as_html", None),
                image_base64=getattr(m, "image_base64", None),
                image_mime_type=getattr(m, "image_mime_type", None),
            ),
        ))
    return out


def insight(els: list[RawElement]) -> Counter:
    return Counter(e.type for e in els)