# api/auth_router.py
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from identity.auth_service import AuthError, login, register

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    company_id: uuid.UUID
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", status_code=201)
async def register_route(body: RegisterRequest, db: AsyncSession = Depends(get_db)) -> dict:
    try:
        user = await register(db, body.company_id, body.email, body.password)
    except AuthError as exc:
        raise HTTPException(409, str(exc))
    return {"id": str(user.id), "email": user.email}


@router.post("/login", response_model=TokenResponse)
async def login_route(body: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    try:
        token = await login(db, body.email, body.password)
    except AuthError as exc:
        raise HTTPException(401, str(exc))
    return TokenResponse(access_token=token)
