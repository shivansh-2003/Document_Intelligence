# api/membership_router.py
#
# ponytail: no access gate on this route -- the first member of a new department
# can't pass require_dept_access (there's no membership yet to grant one). A real
# company-admin/bootstrap concept would close this; out of scope through Phase 3.
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from identity.membership_service import add_member
from models import Department, Role, User

router = APIRouter(prefix="/departments/{dept_id}/members", tags=["memberships"])


class MembershipRequest(BaseModel):
    user_id: uuid.UUID
    role: Role = Role.VIEWER


@router.post("", status_code=201)
async def add_member_route(
    dept_id: uuid.UUID, body: MembershipRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    # Checked separately, not left to the FK violation, so a bad id gets a precise
    # 404 naming which reference is wrong instead of an unhandled 500.
    if await db.get(Department, dept_id) is None:
        raise HTTPException(404, "department not found")
    if await db.get(User, body.user_id) is None:
        raise HTTPException(404, "user not found")

    try:
        membership = await add_member(db, body.user_id, dept_id, body.role)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, "membership already exists")
    return {
        "user_id": str(membership.user_id),
        "dept_id": str(membership.dept_id),
        "role": membership.role.value,
    }
