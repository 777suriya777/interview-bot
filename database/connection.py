"""
Async database connection factory.

Usage in a FastAPI service:
    from database.connection import get_db_session

    @app.get("/example")
    async def example(db: AsyncSession = Depends(get_db_session)):
        result = await db.execute(select(User))
        ...

Services that need asyncpg directly (e.g. adaptive_engine for raw SQL):
    from database.connection import get_asyncpg_pool
"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ── Engine ─────────────────────────────────────────────────────────────

def _build_async_url() -> str:
    """
    Read DATABASE_URL from env and convert to asyncpg-compatible scheme.
    Supports both:
      postgresql://...       → replaced with postgresql+asyncpg://
      postgresql+asyncpg://  → used as-is
    """
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


# Create engine once at module level.
# pool_size=10 handles 50 concurrent sessions with headroom.
# echo=False in prod; set DATABASE_ECHO=true in .env for SQL logging.
engine = create_async_engine(
    _build_async_url(),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,          # test connection before checkout (handles pg restart)
    echo=os.environ.get("DATABASE_ECHO", "false").lower() == "true",
)

# Session factory — expire_on_commit=False avoids lazy-load errors after commit
AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── FastAPI dependency ──────────────────────────────────────────────────

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a session and ensures it's closed.
    Rolls back automatically on exception.

    Example:
        @app.post("/sessions")
        async def create_session(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── asyncpg raw pool (for services that bypass SQLAlchemy ORM) ─────────

import asyncpg  # noqa: E402 — import after stdlib

_asyncpg_pool: asyncpg.Pool | None = None


async def get_asyncpg_pool() -> asyncpg.Pool:
    """
    Return a shared asyncpg connection pool.
    Call this from a FastAPI lifespan startup handler.

    Example in main.py:
        from contextlib import asynccontextmanager
        from database.connection import get_asyncpg_pool, close_asyncpg_pool

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await get_asyncpg_pool()   # warm up pool
            yield
            await close_asyncpg_pool()
    """
    global _asyncpg_pool
    if _asyncpg_pool is None:
        # asyncpg uses postgresql:// scheme (not postgresql+asyncpg://)
        raw_url = os.environ.get("DATABASE_URL", "")
        if not raw_url:
            raise RuntimeError("DATABASE_URL environment variable is not set")
        # Ensure we're using the plain postgresql:// scheme for asyncpg
        asyncpg_url = raw_url.replace("postgresql+asyncpg://", "postgresql://")
        _asyncpg_pool = await asyncpg.create_pool(
            asyncpg_url,
            min_size=2,
            max_size=10,
            command_timeout=30,   # seconds before a query times out
        )
    return _asyncpg_pool


async def close_asyncpg_pool() -> None:
    """Gracefully close the asyncpg pool. Call from lifespan shutdown."""
    global _asyncpg_pool
    if _asyncpg_pool is not None:
        await _asyncpg_pool.close()
        _asyncpg_pool = None
