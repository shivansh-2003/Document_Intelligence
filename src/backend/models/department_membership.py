# models/department_membership.py
import enum
import uuid

from sqlalchemy import Enum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class Role(str, enum.Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


class DepartmentMembership(Base):
    """The actual RBAC join: user_id x dept_id -> role. Composite PK, not a surrogate
    id -- one role per user per department, not a list."""
    __tablename__ = "department_memberships"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), primary_key=True)
    dept_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("departments.id"), primary_key=True)
    role: Mapped[Role] = mapped_column(Enum(Role, name="role_enum"), nullable=False)
