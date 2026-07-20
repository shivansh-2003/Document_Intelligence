# api/parsing_router.py
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from services.parsing_service import AUDIO_TYPES, DOC_TYPES, ParsedDocument, parse_audio_stream, parse_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/parse", tags=["parsing"])

MAX_BYTES = 50 * 1024 * 1024         # ponytail: hard cap, stream to disk if you need bigger
MAX_AUDIO_BYTES = 500 * 1024 * 1024  # audio files run much larger than documents


# ── existing synchronous path: whole file in, whole ParsedDocument out ───────

@router.post("", response_model=ParsedDocument)
async def parse_upload(file: UploadFile = File(...)) -> ParsedDocument:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in DOC_TYPES:
        raise HTTPException(400, f"unsupported type {suffix!r}; allowed: {sorted(DOC_TYPES)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file > {MAX_BYTES // 1024 // 1024} MB")

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        doc = parse_document(tmp.name, doc_id="doc_" + uuid4().hex[:12], dept_id="dept_default")

    doc.source = file.filename           # keep the user's name, not the temp one
    return doc


# ── new: audio path, streamed as SSE ──────────────────────────────────────
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
    n = 0
    try:
        async for chunk in parse_audio_stream(path, doc_id=doc_id, dept_id=dept_id, source_name=source_name):
            n += 1
            yield _sse("chunk", chunk.model_dump())
        yield _sse("done", {"doc_id": doc_id, "chunk_count": n})
    except Exception as exc:
        logger.exception("audio stream failed doc_id=%s dept_id=%s", doc_id, dept_id)
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
    dept_id: str = Form(...),
    doc_id: str | None = Form(None),
) -> StreamingResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_TYPES:
        raise HTTPException(400, f"unsupported audio type {suffix!r}; allowed: {sorted(AUDIO_TYPES)}")
    if not dept_id or not dept_id.strip():
        raise HTTPException(400, "dept_id is required")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"file > {MAX_AUDIO_BYTES // 1024 // 1024} MB")

    # delete=False: the generator, not this handler, owns cleanup -- see the
    # finally block in _audio_event_stream above.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()

    doc_id = doc_id or ("doc_" + uuid4().hex[:12])
    logger.info("Starting audio stream %r doc_id=%s dept_id=%s", file.filename, doc_id, dept_id)

    return StreamingResponse(
        _audio_event_stream(tmp.name, doc_id, dept_id, file.filename or Path(tmp.name).name),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # disable nginx buffering
    )