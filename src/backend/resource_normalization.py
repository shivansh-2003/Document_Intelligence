"""
resource_normalization.py
─────────────────────────
Single-entry orchestration layer: file / URL → NormalizedResource.

Model policy
────────────
One model for everything: qwen3.6
  - qwen3.6:27b  (17 GB, Text + Image, 256K context) — recommended default
  - qwen3.6:35b  (24 GB, Text + Image, 256K context) — higher quality
  - qwen3.6      (resolves to :35b on Ollama latest tag)

qwen3.6 is natively multimodal: the same model handles table summarization
(text-only) and image analysis (vision) — no need for separate model configs.

Flow
────
    file / URL
        │
        ▼
    ResourceNormalizer.run()
        │
        ├─ 1. Detect source_type from extension or URL flag
        ├─ 2. Parse → ParsedDocument (texts, tables, images)
        ├─ 3. Build doc_meta
        ├─ 4. Chunk three streams independently
        │       ├── chunk_text()     → list[TextChunk]
        │       ├── process_tables() → list[TableChunk]
        │       └── process_images() → list[ImageChunk]
        └─ 5. Return NormalizedResource

No embedding or vector-store calls happen here.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from parsing.pdf_parser   import PDFParser
from parsing.word_parser  import WordParser
from parsing.pptx_parser  import PPTXParser
from parsing.text_parser  import TextParser
from parsing.web_parser   import WebParser

from pipelines.text_chunker   import chunk_text,      TextChunk
from pipelines.table_pipeline import process_tables,  TableChunk
from pipelines.image_pipeline import process_images,  ImageChunk


EXTENSION_MAP: dict[str, str] = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".doc":  "docx",
    ".pptx": "pptx",
    ".ppt":  "pptx",
    ".txt":  "txt",
    ".md":   "txt",
    ".text": "txt",
}

SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())
DEFAULT_MODEL = "qwen3.6"


@dataclass
class NormalizationStats:
    doc_type:          str
    source_type:       str
    text_block_count:  int
    table_count:       int
    image_count:       int
    text_chunk_count:  int
    table_chunk_count: int
    image_chunk_count: int
    total_chunks:      int


@dataclass
class NormalizedResource:
    doc_meta:     dict[str, Any]
    text_chunks:  list[TextChunk]  = field(default_factory=list)
    table_chunks: list[TableChunk] = field(default_factory=list)
    image_chunks: list[ImageChunk] = field(default_factory=list)
    stats:        NormalizationStats | None = None

    @property
    def all_chunks(self) -> list:
        return self.text_chunks + self.table_chunks + self.image_chunks


class ResourceNormalizer:
    """
    Stateless orchestrator. Safe to instantiate once and reuse across requests.

    run() args
    ----------
    file_path     : Path to uploaded file. Mutually exclusive with url.
    url           : Web URL to crawl. Mutually exclusive with file_path.
    dept_id       : Department UUID string. Required for RBAC filtering.
    created_by    : User ID of the uploader.
    doc_id        : Optional pre-generated document UUID (auto-generated if omitted).
    model         : Ollama model tag for ALL LLM + VLM tasks.
                    Default "qwen3.6". Use "qwen3.6:27b" for smaller footprint.
    use_llm       : If True, generate LLM summaries for tables. Default True.
    use_vlm       : If True, run VLM analysis on images. Default True.
    extract_images: Whether to extract images during PDF parsing. Default True.
    """

    @staticmethod
    def _detect_source_type(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext not in EXTENSION_MAP:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return EXTENSION_MAP[ext]

    @staticmethod
    def _parse_file(file_path: Path, source_type: str, extract_images: bool):
        if source_type == "pdf":
            return PDFParser(extract_images=extract_images, strategy="hi_res").parse(file_path)
        if source_type == "docx":
            return WordParser().parse(file_path)
        if source_type == "pptx":
            return PPTXParser().parse(file_path)
        if source_type == "txt":
            return TextParser().parse(file_path)
        raise ValueError(f"No parser for source_type='{source_type}'")

    @staticmethod
    def _parse_url(url: str):
        return WebParser(extract_images=True, use_pruning_filter=True).parse(url)

    @staticmethod
    def _build_doc_meta(*, doc_id, dept_id, filename, created_by, upload_date) -> dict:
        return {
            "doc_id": doc_id, "dept_id": dept_id,
            "filename": filename, "upload_date": upload_date, "created_by": created_by,
        }

    def run(
        self,
        *,
        file_path:      str | Path | None = None,
        url:            str | None = None,
        dept_id:        str,
        created_by:     str,
        doc_id:         str | None = None,
        model:          str = DEFAULT_MODEL,
        use_llm:        bool = True,
        use_vlm:        bool = True,
        extract_images: bool = True,
    ) -> NormalizedResource:

        if file_path is None and url is None:
            raise ValueError("Provide either file_path or url.")
        if file_path is not None and url is not None:
            raise ValueError("Provide file_path OR url, not both.")
        if not dept_id or not dept_id.strip():
            raise ValueError("dept_id is required.")
        if not created_by or not created_by.strip():
            raise ValueError("created_by is required.")

        doc_id      = doc_id or ("doc_" + uuid.uuid4().hex[:12])
        upload_date = str(date.today())

        if file_path is not None:
            file_path   = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            source_type = self._detect_source_type(file_path)
            filename    = file_path.name
            logger.info("Parsing file: %r source_type=%r doc_id=%r extract_images=%s",
                        filename, source_type, doc_id, extract_images)
            parsed      = self._parse_file(file_path, source_type, extract_images)
        else:
            source_type = "url"
            filename    = url
            logger.info("Crawling URL: %r doc_id=%r", url, doc_id)
            parsed      = self._parse_url(url)

        doc_meta = self._build_doc_meta(
            doc_id=doc_id, dept_id=dept_id,
            filename=filename, created_by=created_by, upload_date=upload_date,
        )

        raw_texts  = parsed.texts  or []
        raw_tables = parsed.tables or []
        raw_images = parsed.images or []
        logger.info("Parsed %d text blocks, %d tables, %d images from %r",
                    len(raw_texts), len(raw_tables), len(raw_images), filename)

        logger.info("Chunking text blocks (source_type=%r)", source_type)
        text_chunks = chunk_text(
            texts=raw_texts, source_type=source_type, doc_meta=doc_meta,
        )
        logger.info("Text chunking produced %d chunks", len(text_chunks))

        llm_client = {"backend": "ollama", "model": model} if use_llm else None
        logger.info("Processing %d tables (use_llm=%s model=%r)", len(raw_tables), use_llm, model)
        table_chunks = process_tables(
            tables=raw_tables, source_type=source_type,
            doc_meta=doc_meta, llm_client=llm_client,
        )
        logger.info("Table pipeline produced %d chunks", len(table_chunks))

        vlm_config = {"model": model} if use_vlm else None
        logger.info("Processing %d images (use_vlm=%s model=%r)", len(raw_images), use_vlm, model)
        image_chunks = process_images(
            images=raw_images, source_type=source_type,
            doc_meta=doc_meta, vlm_config=vlm_config,
        )
        logger.info("Image pipeline produced %d chunks", len(image_chunks))

        stats = NormalizationStats(
            doc_type          = file_path.suffix.lstrip(".").lower() if file_path else "url",
            source_type       = source_type,
            text_block_count  = len(raw_texts),
            table_count       = len(raw_tables),
            image_count       = len(raw_images),
            text_chunk_count  = len(text_chunks),
            table_chunk_count = len(table_chunks),
            image_chunk_count = len(image_chunks),
            total_chunks      = len(text_chunks) + len(table_chunks) + len(image_chunks),
        )
        logger.info("NormalizedResource ready: doc_id=%r total_chunks=%d", doc_id, stats.total_chunks)

        return NormalizedResource(
            doc_meta=doc_meta,
            text_chunks=text_chunks, table_chunks=table_chunks, image_chunks=image_chunks,
            stats=stats,
        )