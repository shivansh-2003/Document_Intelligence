# identity/auth_service.py — login, register, token issuance. No RBAC here; this
# file doesn't know what a department is.
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import create_access_token, hash_password, verify_password
from models import User


class AuthError(Exception):
    """Bad credentials, inactive user, or duplicate registration. Routers translate
    this to a 401/409 -- this module stays framework-agnostic."""


async def register(db: AsyncSession, company_id: uuid.UUID, email: str, password: str) -> User:
    existing = await db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise AuthError("email already registered")
    user = User(company_id=company_id, email=email, hashed_password=hash_password(password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def login(db: AsyncSession, email: str, password: str) -> str:
    """Returns a signed access token."""
    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(password, user.hashed_password):
        raise AuthError("invalid email or password")
    return create_access_token(str(user.id))
