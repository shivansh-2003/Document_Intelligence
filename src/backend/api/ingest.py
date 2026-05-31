from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from resource_normalization import (
    NormalizedResource,
    ResourceNormalizer,
    SUPPORTED_EXTENSIONS,
)

ingest_router = APIRouter()
_normalizer   = ResourceNormalizer()   # stateless; safe to share


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class IngestStats(BaseModel):
    doc_type:          str
    source_type:       str
    text_block_count:  int
    table_count:       int
    image_count:       int
    text_chunk_count:  int
    table_chunk_count: int
    image_chunk_count: int
    total_chunks:      int


class IngestResponse(BaseModel):
    """
    Full normalized output returned by both /ingest/file and /ingest/url.

    Fields
    ──────
    doc_meta       — doc_id, dept_id, filename, upload_date, created_by
    stats          — parse + chunk counts
    text_chunks    — list of text chunk payloads (id + text + metadata)
    table_chunks   — list of table chunk payloads
    image_chunks   — list of image chunk payloads
    message        — human-readable summary
    """
    doc_meta:     dict[str, Any]
    stats:        IngestStats
    text_chunks:  list[dict[str, Any]] = Field(default_factory=list)
    table_chunks: list[dict[str, Any]] = Field(default_factory=list)
    image_chunks: list[dict[str, Any]] = Field(default_factory=list)
    message:      str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_normalized(resource: NormalizedResource) -> IngestResponse:
    """Convert a NormalizedResource into a JSON-serializable IngestResponse."""

    def _chunk_to_dict(chunk) -> dict[str, Any]:
        """
        Convert any chunk dataclass (TextChunk / TableChunk / ImageChunk)
        to a plain dict.  We include every field for maximum downstream
        flexibility — the embedding stage will pick what it needs.
        """
        d: dict[str, Any] = {}
        for attr in ("chunk_id", "text", "context_text",
                     "chunk_type", "llm_summary",
                     "image_type", "ocr_text"):
            if hasattr(chunk, attr):
                d[attr] = getattr(chunk, attr)
        if hasattr(chunk, "metadata"):
            d["metadata"] = chunk.metadata
        return d

    stats  = resource.stats
    s_dict = IngestStats(
        doc_type          = stats.doc_type,
        source_type       = stats.source_type,
        text_block_count  = stats.text_block_count,
        table_count       = stats.table_count,
        image_count       = stats.image_count,
        text_chunk_count  = stats.text_chunk_count,
        table_chunk_count = stats.table_chunk_count,
        image_chunk_count = stats.image_chunk_count,
        total_chunks      = stats.total_chunks,
    )

    message = (
        f"Normalized '{resource.doc_meta.get('filename', 'resource')}' → "
        f"{stats.total_chunks} chunks "
        f"({stats.text_chunk_count} text · "
        f"{stats.table_chunk_count} table · "
        f"{stats.image_chunk_count} image)"
    )

    return IngestResponse(
        doc_meta     = resource.doc_meta,
        stats        = s_dict,
        text_chunks  = [_chunk_to_dict(c) for c in resource.text_chunks],
        table_chunks = [_chunk_to_dict(c) for c in resource.table_chunks],
        image_chunks = [_chunk_to_dict(c) for c in resource.image_chunks],
        message      = message,
    )


def _llm_client_from_str(value: str | None) -> Any:
    """
    Parse the llm_client form field.

    Accepted values:
        ""  / None / "none"  → None   (skip table summarization)
        "ollama"             → "ollama" (default model qwen3)
    """
    if not value or value.strip().lower() in ("", "none"):
        return None
    if value.strip().lower() == "ollama":
        return "ollama"
    return None


def _vlm_config_from_str(value: str | None) -> dict | None:
    """
    Parse the vlm_config form field.

    Accepted values:
        "" / None / "none"    → None (skip VLM; images get OCR-only analysis)
        "ollama"              → default {"vision_model":"qwen2.5-vl","text_model":"qwen3"}
    """
    if not value or value.strip().lower() in ("", "none"):
        return None
    if value.strip().lower() == "ollama":
        return {"vision_model": "qwen2.5-vl", "text_model": "qwen3"}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@ingest_router.post(
    "/file",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest an uploaded file",
    description=(
        "Upload a file (PDF, DOCX, PPTX, TXT, MD) and receive back all "
        "parsed, chunked, and metadata-tagged content ready for embedding. "
        "Supported extensions: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
    ),
)
async def ingest_file(
    file: UploadFile = File(..., description="File to ingest"),
    dept_id: str = Form(..., description="Department UUID (string)"),
    created_by: str = Form(..., description="User ID of the uploader"),
    doc_id: str | None = Form(
        default=None,
        description="Optional pre-generated document UUID. Auto-generated if omitted.",
    ),
    llm_client: str | None = Form(
        default=None,
        description=(
            "LLM backend for table summarization. "
            "'ollama' → Ollama qwen3.  Leave blank to skip summaries."
        ),
    ),
    vlm_config: str | None = Form(
        default=None,
        description=(
            "VLM backend for image analysis. "
            "'ollama' → qwen2.5-vl + qwen3.  Leave blank for OCR-only fallback."
        ),
    ),
    extract_images: bool = Form(
        default=True,
        description="Whether to extract images from the file (PDF only).",
    ),
) -> IngestResponse:

    # ── 1. Validate file type ────────────────────────────────────────────────
    filename = file.filename or ""
    ext      = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # ── 2. Save upload to a temp file ────────────────────────────────────────
    tmp_dir  = tempfile.mkdtemp(prefix="ingest_")
    tmp_path = Path(tmp_dir) / filename
    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    # ── 3. Normalize ─────────────────────────────────────────────────────────
    try:
        resource = _normalizer.run(
            file_path      = tmp_path,
            dept_id        = dept_id.strip(),
            created_by     = created_by.strip(),
            doc_id         = doc_id.strip() if doc_id else None,
            llm_client     = _llm_client_from_str(llm_client),
            vlm_config     = _vlm_config_from_str(vlm_config),
            extract_images = extract_images,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Normalization failed. Check server logs for details.\n"
                + traceback.format_exc()
            ),
        )
    finally:
        # Always clean up the temp file
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return _serialize_normalized(resource)


# ─────────────────────────────────────────────────────────────────────────────

@ingest_router.post(
    "/url",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a web URL",
    description=(
        "Crawl a URL (supports JS-rendered pages via Crawl4AI) and return "
        "all parsed, chunked, and metadata-tagged content ready for embedding."
    ),
)
async def ingest_url(
    url: str = Form(..., description="Web URL to crawl and ingest"),
    dept_id: str = Form(..., description="Department UUID (string)"),
    created_by: str = Form(..., description="User ID of the uploader"),
    doc_id: str | None = Form(
        default=None,
        description="Optional pre-generated document UUID. Auto-generated if omitted.",
    ),
    llm_client: str | None = Form(
        default=None,
        description="LLM backend for table summarization. 'ollama' or leave blank.",
    ),
    vlm_config: str | None = Form(
        default=None,
        description="VLM backend for image analysis. 'ollama' or leave blank.",
    ),
) -> IngestResponse:

    # ── Basic URL validation ─────────────────────────────────────────────────
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must start with http:// or https://",
        )

    # ── Normalize ────────────────────────────────────────────────────────────
    try:
        resource = _normalizer.run(
            url        = url,
            dept_id    = dept_id.strip(),
            created_by = created_by.strip(),
            doc_id     = doc_id.strip() if doc_id else None,
            llm_client = _llm_client_from_str(llm_client),
            vlm_config = _vlm_config_from_str(vlm_config),
        )
    except (ValueError,) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        # WebParser raises RuntimeError on crawl failure
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to crawl URL: {exc}",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Normalization failed. Check server logs for details.\n"
                + traceback.format_exc()
            ),
        )

    return _serialize_normalized(resource)


# ─────────────────────────────────────────────────────────────────────────────
# Health / supported types
# ─────────────────────────────────────────────────────────────────────────────

@ingest_router.get(
    "/supported-types",
    summary="List supported file types",
    description="Returns the file extensions accepted by the /ingest/file endpoint.",
)
async def supported_types() -> JSONResponse:
    return JSONResponse({
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "url_ingestion":        True,
    })