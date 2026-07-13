# api/parsing_router.py
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.parsing_service import ParsedDocument, parse_document

router = APIRouter(prefix="/parse", tags=["parsing"])

ALLOWED = {".pdf", ".docx", ".pptx"}
MAX_BYTES = 50 * 1024 * 1024   # ponytail: hard cap, stream to disk if you need bigger


@router.post("", response_model=ParsedDocument)
async def parse_upload(file: UploadFile = File(...)) -> ParsedDocument:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"unsupported type {suffix!r}; allowed: {sorted(ALLOWED)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file > {MAX_BYTES // 1024 // 1024} MB")

    # unstructured wants a path, not bytes
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        doc = parse_document(tmp.name)

    doc.source = file.filename           # keep the user's name, not the temp one
    return doc