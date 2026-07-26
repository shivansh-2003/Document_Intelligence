"""api/parsing_router.py — three ways to turn a file into chunks.

  POST /parse               one doc file, synchronous, returns the parsed result
  POST /parse/audio/stream  one audio file, SSE -- chunks arrive as Whisper transcribes
  POST /parse/batch         many files (mixed formats), processed concurrently in the
                             background -- returns immediately, poll status via
                             GET /departments/{dept_id}/documents/{doc_id}
"""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import SessionLocal, get_db
from identity.rbac_service import require_dept_access_form
from indexing.indexing_pipeline import index_chunks
from models import DepartmentMembership, Document, Role
from services.parsing_service import AUDIO_TYPES, DOC_TYPES, ParsedDocument, parse_audio_stream, parse_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parse", tags=["parsing"])

MAX_BYTES = 50 * 1024 * 1024          # doc files: pdf/docx/pptx/txt/md/html
MAX_AUDIO_BYTES = 500 * 1024 * 1024   # audio runs much larger than documents


# ── shared helpers, used by all three routes below ──────────────────────────

def _new_doc_id() -> str:
    return "doc_" + uuid4().hex[:12]


def _classify(filename: str | None) -> tuple[str, str, bool]:
    """filename -> (suffix, doc_type, is_audio). Raises 400 if the extension is
    neither a known document type nor a known audio type."""
    suffix = Path(filename or "").suffix.lower()
    if suffix in AUDIO_TYPES:
        return suffix, "audio", True
    if suffix in DOC_TYPES:
        return suffix, DOC_TYPES[suffix], False
    allowed = sorted(DOC_TYPES) + sorted(AUDIO_TYPES)
    raise HTTPException(400, f"unsupported type {suffix!r} for {filename!r}; allowed: {allowed}")


async def _read_validated(file: UploadFile, max_bytes: int) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, f"empty file: {file.filename!r}")
    if len(data) > max_bytes:
        raise HTTPException(413, f"{file.filename!r} exceeds {max_bytes // 1024 // 1024} MB")
    return data


def _write_temp(data: bytes, suffix: str) -> str:
    """Returns the temp file path. Caller owns cleanup (os.unlink) -- processing
    may outlive the request that wrote this file."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _new_document(
    doc_id: str,
    membership: DepartmentMembership,
    filename: str,
    doc_type: str,
    status: str,
    chunk_count: int | None = None,
) -> Document:
    return Document(
        id=doc_id,
        dept_id=membership.dept_id,
        filename=filename,
        doc_type=doc_type,
        status=status,
        chunk_count=chunk_count,
        uploaded_by=membership.user_id,
    )


async def _mark_document(doc_id: str, **fields) -> None:
    """Opens its own session rather than reusing a request's Depends(get_db) one --
    every caller of this helper (the SSE generator, the batch background task) keeps
    running after its triggering request has already returned, and a request-scoped
    session isn't guaranteed to outlive that."""
    async with SessionLocal() as db:
        document = await db.get(Document, doc_id)
        if document is not None:
            for key, value in fields.items():
                setattr(document, key, value)
            await db.commit()


# ── POST /parse -- one doc file, synchronous ─────────────────────────────────

@router.post("", response_model=ParsedDocument)
async def parse_upload(
    file: UploadFile = File(...),
    membership: DepartmentMembership = Depends(require_dept_access_form(Role.EDITOR)),
    db: AsyncSession = Depends(get_db),
) -> ParsedDocument:
    suffix, doc_type, is_audio = _classify(file.filename)
    if is_audio:
        raise HTTPException(400, f"{file.filename!r} is audio -- use /parse/audio/stream")

    data = await _read_validated(file, MAX_BYTES)
    doc_id = _new_doc_id()
    path = _write_temp(data, suffix)
    try:
        doc = parse_document(path, doc_id=doc_id, dept_id=str(membership.dept_id))
    finally:
        os.unlink(path)

    doc.source = file.filename  # keep the user's name, not the temp one

    # Index before marking ready -- if this raises, the request fails and no Document
    # row gets written, rather than reporting "ready" on a document that isn't
    # actually searchable.
    await index_chunks(doc.chunks)

    db.add(_new_document(doc_id, membership, file.filename, doc_type, "ready", chunk_count=len(doc.chunks)))
    await db.commit()

    return doc


# ── POST /parse/audio/stream -- one audio file, streamed as SSE ─────────────
#
# Audio can't reuse the pattern above: parse_document returns once, after the
# whole file is chunked. Audio wants the opposite -- the caller should see
# chunk 1 while chunk 40 is still transcribing. StreamingResponse + an async
# generator is what makes that visible over plain HTTP; SSE is the framing
# ("event: ...\ndata: ...\n\n") the browser's EventSource understands natively.

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _audio_event_stream(path: str, doc_id: str, dept_id: str, source_name: str) -> AsyncIterator[str]:
    """SSE framing around parse_audio_stream. Kept separate from the route
    handler so it's testable without spinning up FastAPI/ASGI."""
    chunks = []
    try:
        async for chunk in parse_audio_stream(path, doc_id=doc_id, dept_id=dept_id, source_name=source_name):
            chunks.append(chunk)
            yield _sse("chunk", chunk.model_dump())
        await index_chunks(chunks)  # after the client has already seen every chunk
        await _mark_document(doc_id, status="ready", chunk_count=len(chunks))
        yield _sse("done", {"doc_id": doc_id, "chunk_count": len(chunks)})
    except Exception as exc:
        logger.exception("audio stream failed doc_id=%s dept_id=%s", doc_id, dept_id)
        await _mark_document(doc_id, status="failed")
        yield _sse("error", {"doc_id": doc_id, "error": str(exc)})
    finally:
        # The temp file must outlive the whole stream, not just the request
        # handler -- StreamingResponse keeps pulling from this generator
        # after parse_upload_stream() has already returned. Cleanup belongs
        # here, not in the route.
        os.unlink(path)


@router.post("/audio/stream")
async def parse_upload_stream(
    file: UploadFile = File(...),
    doc_id: str | None = Form(None),
    membership: DepartmentMembership = Depends(require_dept_access_form(Role.EDITOR)),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    suffix, doc_type, is_audio = _classify(file.filename)
    if not is_audio:
        raise HTTPException(400, f"{file.filename!r} is not audio -- use /parse")

    data = await _read_validated(file, MAX_AUDIO_BYTES)
    doc_id = doc_id or _new_doc_id()
    path = _write_temp(data, suffix)  # generator, not this handler, owns cleanup -- see finally above
    dept_id = str(membership.dept_id)

    logger.info("Starting audio stream %r doc_id=%s dept_id=%s", file.filename, doc_id, dept_id)

    db.add(_new_document(doc_id, membership, file.filename, doc_type, "processing"))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, f"doc_id {doc_id!r} already exists")

    return StreamingResponse(
        _audio_event_stream(path, doc_id, dept_id, file.filename or Path(path).name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # disable nginx buffering
    )


# ── POST /parse/batch -- many files, mixed formats, processed concurrently ──
#
# BackgroundTasks (FastAPI's built-in) awaits its queued tasks one at a time --
# fine for "don't block the response," wrong for "process N files at once." Plain
# asyncio.create_task() per file is what actually overlaps them: each doc-type file's
# blocking parse call runs in the default thread pool via run_in_executor (same
# reasoning as utils/async_utils.iter_in_thread for audio), so N files' CPU-bound work
# can run at the same time instead of queued behind each other.
#
# ponytail: in-process, no broker -- an in-flight batch is lost on restart, no retry.
# Fine at current scale; upgrade path is backend.md Phase 5 (Celery, separate
# embed/graph_extract queue lanes) once this needs to survive a restart or scale
# across machines.

_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """Fire-and-forget with a strong reference kept until completion -- an
    unreferenced asyncio.Task can be garbage-collected mid-run."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _process_batch_file(doc_id: str, path: str, filename: str, dept_id: str, is_audio: bool) -> None:
    try:
        await _mark_document(doc_id, status="processing")
        if is_audio:
            chunks = [c async for c in parse_audio_stream(path, doc_id=doc_id, dept_id=dept_id, source_name=filename)]
        else:
            loop = asyncio.get_running_loop()
            parsed = await loop.run_in_executor(None, parse_document, path, doc_id, dept_id, filename)
            chunks = parsed.chunks
        await index_chunks(chunks)
        await _mark_document(doc_id, status="ready", chunk_count=len(chunks))
    except Exception:
        logger.exception("batch processing failed doc_id=%s filename=%r", doc_id, filename)
        await _mark_document(doc_id, status="failed")
    finally:
        os.unlink(path)


@router.post("/batch", status_code=202)
async def parse_batch(
    files: list[UploadFile] = File(...),
    membership: DepartmentMembership = Depends(require_dept_access_form(Role.EDITOR)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if not files:
        raise HTTPException(400, "no files provided")

    # Validate everything before writing anything -- a batch either passes whole, or
    # fails fast naming the offending file. Only per-file *processing* failures
    # (parse/LLM errors, handled in _process_batch_file) land in individual
    # status="failed" rows after this point.
    prepared = []  # (filename, suffix, doc_type, is_audio, data)
    for file in files:
        suffix, doc_type, is_audio = _classify(file.filename)
        max_bytes = MAX_AUDIO_BYTES if is_audio else MAX_BYTES
        data = await _read_validated(file, max_bytes)
        prepared.append((file.filename, suffix, doc_type, is_audio, data))

    jobs = []  # (doc_id, tmp_path, filename, is_audio)
    for filename, suffix, doc_type, is_audio, data in prepared:
        doc_id = _new_doc_id()
        path = _write_temp(data, suffix)
        db.add(_new_document(doc_id, membership, filename, doc_type, "pending"))
        jobs.append((doc_id, path, filename, is_audio))

    await db.commit()

    for doc_id, path, filename, is_audio in jobs:
        _spawn(_process_batch_file(doc_id, path, filename, str(membership.dept_id), is_audio))

    return {"jobs": [{"doc_id": doc_id, "filename": filename} for doc_id, _, filename, _ in jobs]}
