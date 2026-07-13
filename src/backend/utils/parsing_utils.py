# cell 2 — schema + common helper. Partition only. No collapsing, no dropping.
from collections import Counter
from pathlib import Path
from pydantic import BaseModel
from unstructured.documents.elements import Element

class RawElement(BaseModel):
    idx: int                        # reading order — the whole point
    category: str                   # raw unstructured category, untouched
    text: str = ""
    html: str | None = None         # Table -> text_as_html
    image_b64: str | None = None    # Image/Table -> base64 payload (pdf)
    page: int | None = None
    source: str

def to_raw(elements: list[Element], path: str) -> list[RawElement]:
    """unstructured Elements -> ordered RawElements. Order preserved, nothing judged."""
    p = Path(path).name
    return [
        RawElement(
            idx=i,
            category=el.category,
            text=(el.text or ""),
            html=getattr(el.metadata, "text_as_html", None),
            image_b64=getattr(el.metadata, "image_base64", None),
            page=getattr(el.metadata, "page_number", None),
            source=p,
        )
        for i, el in enumerate(elements)
    ]

def insight(els: list[RawElement]) -> Counter:
    return Counter(e.category for e in els)   # Title: 208, Table: 72, Image: 3, ...