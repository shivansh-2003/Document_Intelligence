# core/security.py — JWT issuance/verification, password hashing
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from core.config import JWT_ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM, JWT_SECRET_KEY


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_access_token(user_id: str) -> str:
    """Token carries user_id only -- never dept_id/role, so a role change takes
    effect immediately instead of waiting for the token to expire."""
    expires = datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "exp": expires}, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    """Returns user_id. Raises jwt.PyJWTError (expired/invalid) -- caller turns that into a 401."""
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    return payload["sub"]


if __name__ == "__main__":
    token = create_access_token("user_123")
    assert decode_access_token(token) == "user_123"

    hashed = hash_password("hunter2")
    assert verify_password("hunter2", hashed)
    assert not verify_password("wrong", hashed)

    print("core/security.py: self-check passed")
