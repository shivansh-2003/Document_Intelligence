# api/documents_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from identity.rbac_service import require_dept_access
from models import DepartmentMembership, Document, Role
from vectorstore.vector_store import mark_superseded

router = APIRouter(prefix="/departments/{dept_id}/documents", tags=["documents"])


@router.get("/{doc_id}")
async def get_document_status(
    doc_id: str,
    membership: DepartmentMembership = Depends(require_dept_access(Role.VIEWER)),
    db: AsyncSession = Depends(get_db),
) -> dict:
    document = await db.get(Document, doc_id)
    if document is None or document.dept_id != membership.dept_id:
        raise HTTPException(404, "document not found")
    return {
        "id": document.id,
        "filename": document.filename,
        "doc_type": document.doc_type,
        "status": document.status,
        "chunk_count": document.chunk_count,
        "is_current": document.is_current,
        "superseded_by": document.superseded_by,
    }


@router.post("/{doc_id}/supersede/{old_doc_id}", status_code=204)
async def confirm_supersession(
    doc_id: str,
    old_doc_id: str,
    membership: DepartmentMembership = Depends(require_dept_access(Role.EDITOR)),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Explicit, editor-gated confirmation that doc_id supersedes old_doc_id --
    ingestion_versioning.supersession only ever *suggests* this (possible_duplicates
    on the ingest response), it never flips anything itself. See context/retrieval.md §7.2.
    """
    new_doc = await db.get(Document, doc_id)
    old_doc = await db.get(Document, old_doc_id)
    if new_doc is None or old_doc is None or new_doc.dept_id != membership.dept_id or old_doc.dept_id != membership.dept_id:
        raise HTTPException(404, "document not found")

    old_doc.is_current = False
    old_doc.superseded_by = doc_id
    await db.commit()
    await mark_superseded(old_doc_id)  # Qdrant payload flip, no re-upsert
