# api/membership_router.py
#
# ponytail: no access gate on this route -- the first member of a new department
# can't pass require_dept_access (there's no membership yet to grant one). A real
# company-admin/bootstrap concept would close this; out of scope through Phase 3.
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from identity.membership_service import add_member
from models import Role

router = APIRouter(prefix="/departments/{dept_id}/members", tags=["memberships"])


class MembershipRequest(BaseModel):
    user_id: uuid.UUID
    role: Role = Role.VIEWER


@router.post("", status_code=201)
async def add_member_route(
    dept_id: uuid.UUID, body: MembershipRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    membership = await add_member(db, body.user_id, dept_id, body.role)
    return {
        "user_id": str(membership.user_id),
        "dept_id": str(membership.dept_id),
        "role": membership.role.value,
    }
