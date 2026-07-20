# core/database.py
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


# statement_cache_size=0: Neon's pooled endpoint (`-pooler` in the host) runs PgBouncer
# in transaction mode, which doesn't support asyncpg's server-side prepared statements --
# harmless to disable this against a non-pooled DB too, so it's unconditional.
# ssl=True: required by Neon, not implied by the URL scheme for asyncpg the way
# `sslmode=require` works for psycopg2/libpq.
_connect_args: dict = {"statement_cache_size": 0}
if "neon.tech" in DATABASE_URL:
    _connect_args["ssl"] = True

engine = create_async_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
