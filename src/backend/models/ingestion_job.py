# models/ingestion_job.py
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    celery_task: Mapped[str | None] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String, default="parsing")  # parsing -> chunking -> embedding -> indexing
    error: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
