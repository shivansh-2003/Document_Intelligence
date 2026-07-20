# identity/membership_service.py — add/remove a member, change a role, list members.
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import DepartmentMembership, Role


async def add_member(
    db: AsyncSession, user_id: uuid.UUID, dept_id: uuid.UUID, role: Role
) -> DepartmentMembership:
    membership = DepartmentMembership(user_id=user_id, dept_id=dept_id, role=role)
    db.add(membership)
    await db.commit()
    return membership


async def remove_member(db: AsyncSession, user_id: uuid.UUID, dept_id: uuid.UUID) -> None:
    membership = await db.get(DepartmentMembership, (user_id, dept_id))
    if membership is not None:
        await db.delete(membership)
        await db.commit()


async def change_role(
    db: AsyncSession, user_id: uuid.UUID, dept_id: uuid.UUID, role: Role
) -> DepartmentMembership:
    membership = await db.get(DepartmentMembership, (user_id, dept_id))
    if membership is None:
        raise ValueError("no membership to update")
    membership.role = role
    await db.commit()
    return membership


async def list_members(db: AsyncSession, dept_id: uuid.UUID) -> list[DepartmentMembership]:
    result = await db.scalars(
        select(DepartmentMembership).where(DepartmentMembership.dept_id == dept_id)
    )
    return list(result)
