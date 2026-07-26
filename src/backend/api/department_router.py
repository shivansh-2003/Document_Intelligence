# api/department_router.py
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models import Company, Department, IsolationMode

router = APIRouter(prefix="/departments", tags=["departments"])


class DepartmentRequest(BaseModel):
    company_id: uuid.UUID
    name: str
    dept_type: str = "standard"
    regulatory_flags: list[str] = []


@router.post("", status_code=201)
async def create_department(body: DepartmentRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """isolation_mode is hardcoded to LOGICAL here. The real classify() decision
    (regulatory_flags/dept_type -> logical|physical|isolated) belongs to
    vectorstore/resolver.py's HybridResolver, which doesn't exist until Phase 4 --
    this is a placeholder, not the resolver, so collection naming here will need
    recomputing once that lands."""
    # Without this check, an unknown company_id reaches Postgres as a raw FK
    # violation -- an unhandled 500, and the IntegrityError catch below would
    # misreport it as a naming collision instead of the real problem.
    if await db.get(Company, body.company_id) is None:
        raise HTTPException(404, "company not found")

    dept_id = uuid.uuid4()
    department = Department(
        id=dept_id,
        company_id=body.company_id,
        name=body.name,
        dept_type=body.dept_type,
        isolation_mode=IsolationMode.LOGICAL,
        regulatory_flags=body.regulatory_flags,
        qdrant_collection=f"company_{body.company_id}",
        neo4j_label=f"Dept_{dept_id}",
    )
    db.add(department)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, "department collection naming collision")
    await db.refresh(department)
    return {
        "id": str(department.id),
        "name": department.name,
        "isolation_mode": department.isolation_mode.value,
        "qdrant_collection": department.qdrant_collection,
    }
