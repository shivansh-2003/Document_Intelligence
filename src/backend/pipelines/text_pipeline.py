# text_pipeline.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from utils.parsing_utils import RawElement

DROP = {"Header", "Footer", "UncategorizedText"}
TEXT_TYPES = {"NarrativeText", "Formula", "ListItem"}

MAX_CHARS = 2000
OVERLAP = 200


class ChunkMetadata(BaseModel):
    # identity — stamped by parsing_service._tag(), required before indexing
    chunk_id: str | None = None
    doc_id: str | None = None
    dept_id: str | None = None
    doc_type: str | None = None
    # provenance — derived during parsing
    filename: str | None = None
    page_number: int | None = None          # text chunks: page of the Title/first element
    pages: list[int] = []                    # text chunks spanning multiple pages
    element_ids: list[str] = []              # every raw element folded into this chunk
    text_as_html: str | None = None          # table only
    image_base64: str | None = None          # image only
    image_mime_type: str | None = None       # image only


class Chunk(BaseModel):
    kind: str
    idx: int
    title: str | None = None
    text: str
    metadata: ChunkMetadata


def filter_elements(elements: list[RawElement]) -> list[RawElement]:
    return [e for e in elements if e.type not in DROP]


def attach_captions(elements: list[RawElement]) -> list[RawElement]:
    anchors = [e for e in elements if e.type in ("Table", "Image")]
    captions = [e for e in elements if e.type == "FigureCaption"]

    for cap in captions:
        if not anchors:
            break
        nearest = min(anchors, key=lambda a: abs(a.idx - cap.idx))
        nearest.text = f"{cap.text}\n\n{nearest.text}"

    caption_ids = {c.element_id for c in captions}
    return [e for e in elements if e.element_id not in caption_ids]


def _table_image_chunk(el: RawElement) -> Chunk:
    m = el.metadata
    return Chunk(
        kind=el.type.lower(),
        idx=el.idx,
        text=el.text,
        metadata=ChunkMetadata(
            filename=m.filename,
            page_number=m.page_number,
            pages=[m.page_number] if m.page_number is not None else [],
            element_ids=[el.element_id],
            text_as_html=m.text_as_html,
            image_base64=m.image_base64,
            image_mime_type=m.image_mime_type,
        ),
    )


def group_by_title(elements: list[RawElement]) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf_title, buf_parts, buf_idx = None, [], None
    buf_filename, buf_pages, buf_ids = None, [], []

    def flush():
        if buf_parts:
            chunks.append(Chunk(
                kind="text", idx=buf_idx, title=buf_title, text=" ".join(buf_parts),
                metadata=ChunkMetadata(
                    filename=buf_filename,
                    page_number=buf_pages[0] if buf_pages else None,
                    # Drop None before sorting: txt/md/html/url have no pagination, so
                    # every page_number is None. sorted({None}) is a silent [None] that
                    # fails list[int] validation; a mixed [None, 3] raises TypeError.
                    pages=sorted({p for p in buf_pages if p is not None}),
                    element_ids=buf_ids,
                ),
            ))

    for el in elements:
        if el.type in ("Table", "Image"):
            flush()
            buf_title, buf_parts, buf_idx, buf_pages, buf_ids = None, [], None, [], []
            chunks.append(_table_image_chunk(el))
            continue

        if el.type == "Title":
            flush()
            buf_title, buf_parts, buf_idx = el.text, [el.text], el.idx
            buf_filename, buf_pages, buf_ids = el.metadata.filename, [el.metadata.page_number], [el.element_id]
        elif el.type in TEXT_TYPES:
            if buf_idx is None:
                buf_idx, buf_filename = el.idx, el.metadata.filename
            buf_parts.append(el.text)
            buf_pages.append(el.metadata.page_number)
            buf_ids.append(el.element_id)

    flush()
    return sorted(chunks, key=lambda c: c.idx)


def split_oversized(chunks: list[Chunk], max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[Chunk]:
    """Size ceiling for text chunks.

    group_by_title relies on Title cadence to keep chunks embeddable. A .txt file can
    contain zero detected Titles, in which case every NarrativeText folds into one
    buffer and you get a single 200k-char chunk the embedder silently truncates.
    No-op for well-structured PDF/DOCX; the safety net for txt/md/html.

    Sub-chunks inherit the parent's idx (sort is stable, so order holds), title, and
    element_ids — citation tracking still resolves to the source elements.
    Tables and images are never split: a half table is not a table.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)
    out: list[Chunk] = []
    for c in chunks:
        if c.kind != "text" or len(c.text) <= max_chars:
            out.append(c)
            continue
        out.extend(c.model_copy(update={"text": part}, deep=True)
                   for part in splitter.split_text(c.text))
    return out


def normalize_document(elements: list[RawElement]) -> list[Chunk]:
    els = filter_elements(elements)
    els = attach_captions(els)
    return split_oversized(group_by_title(els))