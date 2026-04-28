"""Async SQLAlchemy engine and session helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from rust_assistant.infrastructure.adapters.sqlalchemy.config import SqlAlchemyConfig


AsyncSessionFactory = async_sessionmaker[AsyncSession]


def build_async_engine(config: SqlAlchemyConfig) -> Optional[AsyncEngine]:
    """Build an async SQLAlchemy engine when a database URL is configured."""
    if not config.url:
        return None

    return create_async_engine(
        config.url,
        echo=config.echo,
        pool_pre_ping=True,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
    )


def build_session_factory(engine: Optional[AsyncEngine]) -> Optional[AsyncSessionFactory]:
    """Create an async session factory for the provided engine."""
    if engine is None:
        return None

    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def database_is_ready(session_factory: Optional[AsyncSessionFactory]) -> bool:
    """Check whether the configured database accepts a simple query."""
    if session_factory is None:
        return False

    try:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


async def get_db_session_context(
    session_factory: Optional[AsyncSessionFactory]
) -> AsyncIterator[Optional[AsyncSession]]:
    """Yield a request-scoped async database session when available."""
    if session_factory is None:
        yield None
        return

    async with session_factory() as session:
        yield session


async def dispose_engine(engine: Optional[AsyncEngine]) -> None:
    """Dispose the async engine during application shutdown."""
    if engine is not None:
        await engine.dispose()
