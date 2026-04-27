"""SQLAlchemy-backed unit of work implementation."""

from __future__ import annotations

from types import TracebackType
from typing import Optional, Type

from sqlalchemy.ext.asyncio import AsyncSession

from rust_assistant.application.ports.repositories.chunk_repository import ChunkRepository
from rust_assistant.application.ports.repositories.document_repository import DocumentRepository
from rust_assistant.application.ports.uow import UnitOfWork
from rust_assistant.infrastructure.outbound.sqlalchemy.repositories.chunk_repository import (
    SqlAlchemyChunkRepository,
)
from rust_assistant.infrastructure.outbound.sqlalchemy.repositories.document_repository import (
    SqlAlchemyDocumentRepository,
)
from rust_assistant.infrastructure.outbound.sqlalchemy.session import AsyncSessionFactory


class SqlAlchemyUnitOfWork(UnitOfWork):
    """Own one async SQLAlchemy session and expose repositories bound to that session."""

    def __init__(self, session_factory: AsyncSessionFactory):
        self._session_factory = session_factory
        self._session: Optional[AsyncSession] = None
        self.documents: DocumentRepository
        self.chunks: ChunkRepository

    async def __aenter__(self) -> "SqlAlchemyUnitOfWork":
        self._session = self._session_factory()
        self.documents = SqlAlchemyDocumentRepository(self._session)
        self.chunks = SqlAlchemyChunkRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        if self._session is None:
            return None

        try:
            if exc_type is not None:
                await self.rollback()
        finally:
            await self._session.close()
            self._session = None
        return None

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("UnitOfWork session is not initialized")
        await self._session.commit()

    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("UnitOfWork session is not initialized")
        await self._session.rollback()
