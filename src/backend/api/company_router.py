# api/company_router.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models import Company

router = APIRouter(prefix="/companies", tags=["companies"])


class CompanyRequest(BaseModel):
    name: str
    slug: str


@router.post("", status_code=201)
async def create_company(body: CompanyRequest, db: AsyncSession = Depends(get_db)) -> dict:
    company = Company(name=body.name, slug=body.slug)
    db.add(company)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, "slug already exists")
    await db.refresh(company)
    return {"id": str(company.id), "name": company.name, "slug": company.slug}
