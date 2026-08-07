# models/document.py
import uuid
from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class Document(Base):
    __tablename__ = "documents"

    # String, not UUID -- caller supplies the same "doc_"-prefixed id already stamped
    # onto every chunk by parsing_service._tag_chunk. One id across Postgres and the
    # chunk metadata, not two disconnected schemes for the same document.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    dept_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("departments.id"))
    filename: Mapped[str | None] = mapped_column(String)
    doc_type: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")
    s3_path: Mapped[str | None] = mapped_column(String)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # §7.2 supersession — set only via the explicit confirm-supersession route
    # (documents_router.py), never inferred automatically from ingest.
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    superseded_by: Mapped[str | None] = mapped_column(ForeignKey("documents.id"))
