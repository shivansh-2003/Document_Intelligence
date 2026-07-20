# identity/rbac_service.py — the single choke point every dept-scoped route imports.
import uuid

import jwt
from fastapi import Depends, Form, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import decode_access_token
from models import DepartmentMembership, Role, User

_ROLE_RANK = {Role.VIEWER: 0, Role.EDITOR: 1, Role.ADMIN: 2}
_bearer = HTTPBearer()


def role_rank(role: Role) -> int:
    return _ROLE_RANK[role]


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    try:
        user_id = decode_access_token(credentials.credentials)
        user_uuid = uuid.UUID(user_id)
    except (jwt.PyJWTError, ValueError):
        raise HTTPException(401, "invalid or expired token")
    user = await db.get(User, user_uuid)
    if user is None or not user.is_active:
        raise HTTPException(401, "user not found or inactive")
    return user


async def _check_access(
    dept_id: uuid.UUID, min_role: Role, user: User, db: AsyncSession
) -> DepartmentMembership:
    membership = await db.get(DepartmentMembership, (user.id, dept_id))
    if membership is None or role_rank(membership.role) < role_rank(min_role):
        raise HTTPException(403, "insufficient department access")
    return membership


def require_dept_access(min_role: Role):
    """Dependency factory: Depends(require_dept_access(Role.EDITOR)).

    dept_id is NOT passed here -- the inner dependency declares a bare `dept_id: uuid.UUID`
    parameter, which FastAPI resolves the same way it resolves any endpoint parameter:
    automatically bound to a path segment when the route uses `/dept/{dept_id}/...`,
    and validated as a UUID before the handler ever runs (bad input -> 422, not a
    hand-written check). That covers every route from Phase 3 onward that takes
    dept_id in the path. For multipart/form routes (file uploads), use
    require_dept_access_form instead -- a bare parameter can't read a form field, and
    a route can't mix Path- and Form-sourced values under the same parameter name.

    Role is looked up fresh from department_memberships on every call, never cached in
    the token, so a role change takes effect on the next request without re-login.
    """

    async def dependency(
        dept_id: uuid.UUID,
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> DepartmentMembership:
        return await _check_access(dept_id, min_role, user, db)

    return dependency


def require_dept_access_form(min_role: Role):
    """Same as require_dept_access, but reads dept_id from a multipart form field --
    for upload routes like /parse and /parse/audio/stream, which take dept_id
    alongside the uploaded file rather than in the path."""

    async def dependency(
        dept_id: uuid.UUID = Form(...),
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> DepartmentMembership:
        return await _check_access(dept_id, min_role, user, db)

    return dependency


if __name__ == "__main__":
    assert role_rank(Role.ADMIN) > role_rank(Role.EDITOR) > role_rank(Role.VIEWER)
    assert role_rank(Role.ADMIN) >= role_rank(Role.ADMIN)  # >=, not == -- admin passes a viewer check
    print("identity/rbac_service.py: self-check passed")
