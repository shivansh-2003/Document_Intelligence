# models/department.py
import enum
import uuid
from datetime import datetime

from sqlalchemy import ARRAY, Enum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class IsolationMode(str, enum.Enum):
    """Set once by vectorstore/resolver.py at provisioning time, never recomputed
    per-request -- see context/backend.md §3."""
    LOGICAL = "logical"
    PHYSICAL = "physical"
    ISOLATED = "isolated"


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    dept_type: Mapped[str] = mapped_column(String, default="standard")
    isolation_mode: Mapped[IsolationMode] = mapped_column(
        Enum(IsolationMode, name="isolation_mode_enum"), default=IsolationMode.LOGICAL
    )
    regulatory_flags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    qdrant_collection: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    neo4j_label: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
