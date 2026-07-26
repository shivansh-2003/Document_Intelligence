# api/documents_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from identity.rbac_service import require_dept_access
from models import DepartmentMembership, Document, Role

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
    }
