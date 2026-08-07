# ingestion_versioning/corpus_version.py — §7.1: atomic per-dept_id counter, bump-on-ingest.
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models import Department


async def bump_corpus_version(db: AsyncSession, dept_id: uuid.UUID) -> int:
    """Postgres's own atomic increment, not hand-rolled locking. Called from
    indexing.indexing_pipeline.index_chunks() right after a successful upsert.
    This counter is the entire tier 2/3 cache invalidation mechanism (baked into
    caching/cache_keys.py's keys) -- a re-ingest naturally misses old cache
    entries, no separate cache-bust step needed."""
    result = await db.execute(
        update(Department)
        .where(Department.id == dept_id)
        .values(corpus_version=Department.corpus_version + 1)
        .returning(Department.corpus_version)
    )
    await db.commit()
    return result.scalar_one()


async def get_versions(db: AsyncSession, dept_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    """dept_ids -> {dept_id: corpus_version}, read by caching/cache_keys.py to
    build tier 2/3 keys."""
    rows = (await db.execute(
        select(Department.id, Department.corpus_version).where(Department.id.in_(dept_ids))
    )).all()
    return dict(rows)
