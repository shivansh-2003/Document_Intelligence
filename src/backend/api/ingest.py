"""
api/ingest.py
─────────────
FastAPI router for the resource ingestion endpoint.

Endpoints
─────────
POST /ingest/file      Upload a file (PDF, DOCX, PPTX, TXT/MD)
POST /ingest/url       Submit a web URL
GET  /ingest/supported-types

Model
─────
Both endpoints use a single qwen3.6 model for all LLM/VLM tasks:
  model param: "qwen3.6" (default) | "qwen3.6:27b" | "qwen3.6:35b"

Mounted in main.py:
    app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
"""

from __future__ import annotations

import logging
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
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)

ingest_router = APIRouter()
_normalizer   = ResourceNormalizer()


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
    doc_meta:     dict[str, Any]
    stats:        IngestStats
    text_chunks:  list[dict[str, Any]] = Field(default_factory=list)
    table_chunks: list[dict[str, Any]] = Field(default_factory=list)
    image_chunks: list[dict[str, Any]] = Field(default_factory=list)
    message:      str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialize(resource: NormalizedResource) -> IngestResponse:
    def _to_dict(chunk) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for attr in ("chunk_id", "text", "context_text",
                     "chunk_type", "llm_summary", "image_type", "ocr_text"):
            if hasattr(chunk, attr):
                d[attr] = getattr(chunk, attr)
        if hasattr(chunk, "metadata"):
            d["metadata"] = chunk.metadata
        return d

    s = resource.stats
    return IngestResponse(
        doc_meta     = resource.doc_meta,
        stats        = IngestStats(
            doc_type=s.doc_type, source_type=s.source_type,
            text_block_count=s.text_block_count, table_count=s.table_count,
            image_count=s.image_count, text_chunk_count=s.text_chunk_count,
            table_chunk_count=s.table_chunk_count, image_chunk_count=s.image_chunk_count,
            total_chunks=s.total_chunks,
        ),
        text_chunks  = [_to_dict(c) for c in resource.text_chunks],
        table_chunks = [_to_dict(c) for c in resource.table_chunks],
        image_chunks = [_to_dict(c) for c in resource.image_chunks],
        message      = (
            f"Normalized '{resource.doc_meta.get('filename', 'resource')}' → "
            f"{s.total_chunks} chunks "
            f"({s.text_chunk_count} text · {s.table_chunk_count} table · "
            f"{s.image_chunk_count} image)"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest/file
# ─────────────────────────────────────────────────────────────────────────────

@ingest_router.post(
    "/file",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest an uploaded file",
    description=(
        "Upload a file (PDF, DOCX, PPTX, TXT, MD) and receive parsed, chunked, "
        "and metadata-tagged content ready for embedding. "
        "Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
    ),
)
async def ingest_file(
    file: UploadFile = File(..., description="File to ingest"),
    dept_id:    str = Form(..., description="Department UUID"),
    created_by: str = Form(..., description="User ID of the uploader"),
    doc_id: str | None = Form(
        default=None,
        description="Optional document UUID. Auto-generated if omitted.",
    ),
    model: str = Form(
        default=DEFAULT_MODEL,
        description=(
            "Ollama model for table summarization and image analysis. "
            f"Default: '{DEFAULT_MODEL}'. Options: qwen3.6:27b, qwen3.6:35b"
        ),
    ),
    use_llm: bool = Form(
        default=True,
        description="Generate LLM summaries for tables (requires Ollama running).",
    ),
    use_vlm: bool = Form(
        default=True,
        description="Run VLM vision analysis on images (requires Ollama running).",
    ),
    extract_images: bool = Form(
        default=True,
        description="Extract images from the file during parsing (PDF only).",
    ),
) -> IngestResponse:

    filename = file.filename or ""
    ext      = Path(filename).suffix.lower()
    logger.info("File ingest request: filename=%r ext=%r dept_id=%r created_by=%r model=%r use_llm=%s use_vlm=%s extract_images=%s",
                filename, ext, dept_id, created_by, model, use_llm, use_vlm, extract_images)

    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning("Rejected unsupported file type %r", ext)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    tmp_dir  = tempfile.mkdtemp(prefix="ingest_")
    tmp_path = Path(tmp_dir) / filename
    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Saved upload to temp path: %s", tmp_path)
    finally:
        file.file.close()

    try:
        logger.info("Starting normalization for %r", filename)
        resource = _normalizer.run(
            file_path      = tmp_path,
            dept_id        = dept_id.strip(),
            created_by     = created_by.strip(),
            doc_id         = doc_id.strip() if doc_id else None,
            model          = model.strip() or DEFAULT_MODEL,
            use_llm        = use_llm,
            use_vlm        = use_vlm,
            extract_images = extract_images,
        )
        s = resource.stats
        logger.info(
            "Normalization complete: doc_id=%r total_chunks=%d "
            "(text=%d table=%d image=%d)",
            resource.doc_meta.get("doc_id"), s.total_chunks,
            s.text_chunk_count, s.table_chunk_count, s.image_chunk_count,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("Validation error for %r: %s", filename, exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception:
        logger.exception("Normalization failed for %r", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Normalization failed.\n" + traceback.format_exc(),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Cleaned up temp dir: %s", tmp_dir)

    return _serialize(resource)


# ─────────────────────────────────────────────────────────────────────────────
# POST /ingest/url
# ─────────────────────────────────────────────────────────────────────────────

@ingest_router.post(
    "/url",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a web URL",
    description="Crawl a URL (JS-rendered pages supported via Crawl4AI) and return normalized chunks.",
)
async def ingest_url(
    url:        str = Form(..., description="Web URL to crawl"),
    dept_id:    str = Form(..., description="Department UUID"),
    created_by: str = Form(..., description="User ID of the uploader"),
    doc_id: str | None = Form(default=None, description="Optional document UUID."),
    model: str = Form(
        default=DEFAULT_MODEL,
        description=f"Ollama model for LLM/VLM tasks. Default: '{DEFAULT_MODEL}'.",
    ),
    use_llm: bool = Form(default=True, description="Generate LLM summaries for tables."),
    use_vlm: bool = Form(default=True, description="Run VLM analysis on images."),
) -> IngestResponse:

    url = url.strip()
    logger.info("URL ingest request: url=%r dept_id=%r created_by=%r model=%r use_llm=%s use_vlm=%s",
                url, dept_id, created_by, model, use_llm, use_vlm)

    if not url.startswith(("http://", "https://")):
        logger.warning("Rejected invalid URL scheme: %r", url)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="URL must start with http:// or https://",
        )

    try:
        logger.info("Starting normalization for URL: %r", url)
        resource = _normalizer.run(
            url        = url,
            dept_id    = dept_id.strip(),
            created_by = created_by.strip(),
            doc_id     = doc_id.strip() if doc_id else None,
            model      = model.strip() or DEFAULT_MODEL,
            use_llm    = use_llm,
            use_vlm    = use_vlm,
        )
        s = resource.stats
        logger.info(
            "URL normalization complete: doc_id=%r total_chunks=%d "
            "(text=%d table=%d image=%d)",
            resource.doc_meta.get("doc_id"), s.total_chunks,
            s.text_chunk_count, s.table_chunk_count, s.image_chunk_count,
        )
    except (ValueError,) as exc:
        logger.warning("Validation error for URL %r: %s", url, exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Crawl failed for URL %r: %s", url, exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Crawl failed: {exc}")
    except Exception:
        logger.exception("Normalization failed for URL %r", url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Normalization failed.\n" + traceback.format_exc(),
        )

    return _serialize(resource)


# ─────────────────────────────────────────────────────────────────────────────
# GET /ingest/supported-types
# ─────────────────────────────────────────────────────────────────────────────

@ingest_router.get("/supported-types", summary="List supported file types")
async def supported_types() -> JSONResponse:
    return JSONResponse({
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "url_ingestion": True,
        "default_model": DEFAULT_MODEL,
        "available_model_tags": ["qwen3.6", "qwen3.6:27b", "qwen3.6:35b"],
    })