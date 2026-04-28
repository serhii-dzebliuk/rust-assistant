"""SQLAlchemy document repository adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.value_objects.identifiers import DocumentId
from rust_assistant.infrastructure.adapters.sqlalchemy.mappers import (
    map_document_from_domain,
    map_document_to_domain,
)
from rust_assistant.infrastructure.adapters.sqlalchemy.models import DocumentRecord


class SqlAlchemyDocumentRepository:
    """Persist and query canonical documents via SQLAlchemy."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, document: Document) -> None:
        """Persist one canonical document via the current unit-of-work session."""

        await self.add_many([document])

    async def add_many(self, documents: Sequence[Document]) -> None:
        """Persist multiple canonical documents via the current unit-of-work session."""

        if not documents:
            return

        self._session.add_all([map_document_from_domain(document) for document in documents])
        await self._session.flush()

    async def get(self, document_id: DocumentId) -> Optional[Document]:
        """Load one canonical document by business UUID."""

        record = await self._session.scalar(
            select(DocumentRecord).where(DocumentRecord.id == document_id)
        )
        if record is None:
            return None
        return map_document_to_domain(record)

    async def delete_all(self) -> None:
        """Delete all documents and let the database cascade to chunks."""

        await self._session.execute(
            delete(DocumentRecord).execution_options(synchronize_session=False)
        )
        await self._session.flush()
