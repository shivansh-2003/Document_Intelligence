"""
resource_normalization.py
─────────────────────────
Single-entry orchestration layer that converts any supported resource
(file upload or URL) into a structured, retrieval-ready payload.

Flow:
    file / URL
        │
        ▼
    ResourceNormalizer.run()
        │
        ├─ 1. Detect doc_type from extension / URL flag
        │
        ├─ 2. Parse  → ParsedDocument  (texts, tables, images)
        │       │
        │       ├── PDFParser      (.pdf)
        │       ├── WordParser     (.docx)
        │       ├── PPTXParser     (.pptx)
        │       ├── TextParser     (.txt / .md)
        │       └── WebParser      (URL)
        │
        ├─ 3. Build doc_meta (doc_id, dept_id, filename, …)
        │
        ├─ 4. Chunk all three content streams independently
        │       ├── chunk_text()     → list[TextChunk]
        │       ├── process_tables() → list[TableChunk]
        │       └── process_images() → list[ImageChunk]
        │
        └─ 5. Return NormalizedResource
                ├── doc_meta
                ├── text_chunks
                ├── table_chunks
                ├── image_chunks
                └── stats (counts + parse summary)

The output of this module is the contract for the embedding stage.
No embedding or vector-store calls happen here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# ── Parsers ──────────────────────────────────────────────────────────────────
from parsing.pdf_parser import PDFParser
from parsing.word_parser import WordParser
from parsing.pptx_parser import PPTXParser
from parsing.text_parser import TextParser
from parsing.web_parser import WebParser

# ── Chunking pipelines ───────────────────────────────────────────────────────
from pipelines.text_chunker import chunk_text, TextChunk
from pipelines.table_pipeline import process_tables, TableChunk
from pipelines.image_pipeline import process_images, ImageChunk


# ─────────────────────────────────────────────────────────────────────────────
# Supported file extensions and their source_type labels
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Output data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizationStats:
    """Lightweight parse + chunk summary — useful for logging and API response."""
    doc_type:          str
    source_type:       str          # "pdf" | "docx" | "pptx" | "txt" | "url"
    text_block_count:  int          # raw text blocks from parser
    table_count:       int          # raw tables from parser
    image_count:       int          # raw images from parser
    text_chunk_count:  int          # chunks produced by text_chunker
    table_chunk_count: int          # chunks produced by table_pipeline
    image_chunk_count: int          # chunks produced by image_pipeline
    total_chunks:      int          # sum of above three


@dataclass
class NormalizedResource:
    """
    Complete normalized output for one document or URL.
    Ready to pass to the embedding stage.
    """
    doc_meta:     dict[str, Any]        # doc_id, dept_id, filename, upload_date, created_by
    text_chunks:  list[TextChunk]  = field(default_factory=list)
    table_chunks: list[TableChunk] = field(default_factory=list)
    image_chunks: list[ImageChunk] = field(default_factory=list)
    stats:        NormalizationStats | None = None

    @property
    def all_chunks(self) -> list:
        """Flat list of every chunk — convenience for the embedding stage."""
        return self.text_chunks + self.table_chunks + self.image_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ResourceNormalizer:
    """
    Stateless orchestrator.  Call .run() once per document.

    Args (passed to run()):
        file_path:    Absolute path to the uploaded file on disk.
                      Mutually exclusive with url.
        url:          Web URL to crawl.
                      Mutually exclusive with file_path.
        dept_id:      Department UUID (string).  Required — used for RBAC
                      filtering at query time.
        created_by:   User ID of the uploader.
        doc_id:       Optional pre-generated document UUID.  Auto-generated
                      if not supplied.
        llm_client:   LLM backend for table summarization.
                      None → skip summaries (fast, no Ollama needed).
                      "ollama" or {"backend":"ollama","model":"qwen3"} → Ollama.
        vlm_config:   VLM backend config for image analysis.
                      None → skip VLM (images get OCR-text-only analysis).
                      {"vision_model":"qwen2.5-vl","text_model":"qwen3"} → Ollama.
        extract_images: Whether to extract images during PDF parsing.
                        Default True.

    Returns:
        NormalizedResource
    """

    # ── Route: extension → parser class ──────────────────────────────────────

    @staticmethod
    def _detect_source_type(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext not in EXTENSION_MAP:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return EXTENSION_MAP[ext]

    @staticmethod
    def _parse_file(file_path: Path, source_type: str, extract_images: bool):
        """Dispatch to the right parser and return a ParsedDocument."""
        if source_type == "pdf":
            return PDFParser(extract_images=extract_images, strategy="hi_res").parse(file_path)
        if source_type == "docx":
            return WordParser().parse(file_path)
        if source_type == "pptx":
            return PPTXParser().parse(file_path)
        if source_type == "txt":
            return TextParser().parse(file_path)
        raise ValueError(f"No parser registered for source_type='{source_type}'")

    @staticmethod
    def _parse_url(url: str):
        """Crawl a URL and return a ParsedWebDocument."""
        parser = WebParser(extract_images=True, use_pruning_filter=True)
        return parser.parse(url)

    # ── doc_meta builder ─────────────────────────────────────────────────────

    @staticmethod
    def _build_doc_meta(
        *,
        doc_id: str,
        dept_id: str,
        filename: str,
        created_by: str,
        upload_date: str,
    ) -> dict[str, Any]:
        return {
            "doc_id":      doc_id,
            "dept_id":     dept_id,
            "filename":    filename,
            "upload_date": upload_date,
            "created_by":  created_by,
        }

    # ── Public entry point ───────────────────────────────────────────────────

    def run(
        self,
        *,
        file_path: str | Path | None = None,
        url: str | None = None,
        dept_id: str,
        created_by: str,
        doc_id: str | None = None,
        llm_client: Any = None,
        vlm_config: dict | None = None,
        extract_images: bool = True,
    ) -> NormalizedResource:
        """
        Main pipeline entry point.

        Exactly one of file_path or url must be provided.
        """
        # ── Validate inputs ──────────────────────────────────────────────────
        if file_path is None and url is None:
            raise ValueError("Provide either file_path or url.")
        if file_path is not None and url is not None:
            raise ValueError("Provide file_path OR url, not both.")
        if not dept_id or not dept_id.strip():
            raise ValueError("dept_id is required and must not be empty.")
        if not created_by or not created_by.strip():
            raise ValueError("created_by is required and must not be empty.")

        # ── Generate stable IDs ──────────────────────────────────────────────
        doc_id       = doc_id or ("doc_" + uuid.uuid4().hex[:12])
        upload_date  = str(date.today())

        # ── Branch: file vs URL ──────────────────────────────────────────────
        if file_path is not None:
            file_path   = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            source_type = self._detect_source_type(file_path)
            filename    = file_path.name
            parsed      = self._parse_file(file_path, source_type, extract_images)

        else:  # URL branch
            source_type = "url"
            filename    = url          # store the URL as the "filename"
            parsed      = self._parse_url(url)

        # ── Build shared doc_meta ────────────────────────────────────────────
        doc_meta = self._build_doc_meta(
            doc_id=doc_id,
            dept_id=dept_id,
            filename=filename,
            created_by=created_by,
            upload_date=upload_date,
        )

        # ── Extract raw content lists ────────────────────────────────────────
        # ParsedDocument (file) and ParsedWebDocument (URL) both expose
        # .texts, .tables, .images — same shape.
        raw_texts  = parsed.texts  if parsed.texts  else []
        raw_tables = parsed.tables if parsed.tables else []
        raw_images = parsed.images if parsed.images else []

        # ── Chunk all three streams ──────────────────────────────────────────
        text_chunks  = chunk_text(
            texts=raw_texts,
            source_type=source_type,
            doc_meta=doc_meta,
        )

        table_chunks = process_tables(
            tables=raw_tables,
            source_type=source_type,
            doc_meta=doc_meta,
            llm_client=llm_client,
        )

        image_chunks = process_images(
            images=raw_images,
            source_type=source_type,
            doc_meta=doc_meta,
            vlm_backend="ollama",
            vlm_config=vlm_config,
        )

        # ── Build stats ──────────────────────────────────────────────────────
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

        return NormalizedResource(
            doc_meta     = doc_meta,
            text_chunks  = text_chunks,
            table_chunks = table_chunks,
            image_chunks = image_chunks,
            stats        = stats,
        )