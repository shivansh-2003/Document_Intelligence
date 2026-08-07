# api/retrieval_router.py — POST /query, cross-department, bearer + viewer.
#
# No path-param dept_id: a query can span every department the user belongs to,
# so this doesn't reuse require_dept_access/require_dept_access_form (both
# single-dept, path/form-bound). retrieval.scope_resolver is what narrows scope
# instead -- see context/retrieval.md §3.3 and §9.
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from identity.rbac_service import get_current_user
from models import User
from services.retrieval_service import RetrievalResponse, answer_query

router = APIRouter(prefix="/query", tags=["retrieval"])


class QueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    dept_hint: list[str] = []


@router.post("", response_model=RetrievalResponse)
async def query_route(
    body: QueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RetrievalResponse:
    return await answer_query(db, user, body.query, body.conversation_id, body.dept_hint)
