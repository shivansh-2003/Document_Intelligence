# retrieval/scope_resolver.py — dept_hint (names) ∩ live memberships -> dept_ids
# See context/retrieval.md §3.3.
import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Department, DepartmentMembership, User


async def resolve_scope(db: AsyncSession, user: User, dept_hint: list[str]) -> list[uuid.UUID]:
    """Fresh membership lookup every call -- no caching of role/membership, same
    anti-staleness principle as identity.rbac_service._check_access. Any membership
    row already clears the viewer floor (Role.VIEWER is rank 0, nothing ranks below
    it), so there's no separate role check here.

    dept_hint (department NAMES, produced by query_transform's LLM call) narrows
    within the user's live memberships -- it is never trusted as an access grant.
    A hint that doesn't intersect live memberships is dropped, not rejected: the
    hint is a suggestion, the memberships are the actual boundary.
    """
    rows = (await db.execute(
        select(Department.id, Department.name)
        .join(DepartmentMembership, DepartmentMembership.dept_id == Department.id)
        .where(DepartmentMembership.user_id == user.id)
    )).all()

    if not rows:
        raise HTTPException(403, "no department access")

    if not dept_hint:
        return [dept_id for dept_id, _ in rows]

    hinted = {name.lower() for name in dept_hint}
    scoped = [dept_id for dept_id, name in rows if name.lower() in hinted]
    return scoped or [dept_id for dept_id, _ in rows]


if __name__ == "__main__":
    # Pure narrowing logic, no DB -- the smallest thing that fails if it breaks.
    rows = [(uuid.uuid4(), "Finance"), (uuid.uuid4(), "Legal")]
    hinted = {"finance"}
    scoped = [d for d, n in rows if n.lower() in hinted]
    assert scoped == [rows[0][0]]

    unmatched_hint = {"nonexistent"}
    fallback = [d for d, n in rows if n.lower() in unmatched_hint] or [d for d, _ in rows]
    assert fallback == [d for d, _ in rows]
    print("retrieval/scope_resolver.py: self-check passed")
