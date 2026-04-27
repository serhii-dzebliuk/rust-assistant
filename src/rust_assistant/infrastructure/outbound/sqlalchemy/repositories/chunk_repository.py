"""SQLAlchemy chunk repository adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from rust_assistant.application.dto.chunk_context import ChunkContext
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.value_objects.identifiers import ChunkId, DocumentId
from rust_assistant.infrastructure.outbound.sqlalchemy.mappers import (
    map_chunk_context_from_record,
    map_chunk_from_domain,
    map_chunk_to_domain,
)
from rust_assistant.infrastructure.outbound.sqlalchemy.models import ChunkRecord, DocumentRecord


class SqlAlchemyChunkRepository:
    """Persist and query canonical chunks via SQLAlchemy."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, chunk: Chunk) -> None:
        """Persist one canonical chunk via the current unit-of-work session."""

        await self.add_many([chunk])

    async def add_many(self, chunks: Sequence[Chunk]) -> None:
        """Persist multiple canonical chunks via the current unit-of-work session."""

        if not chunks:
            return

        document_pks = await self._load_document_pks(chunks)
        self._session.add_all(
            [map_chunk_from_domain(chunk, document_pks[chunk.document_id]) for chunk in chunks]
        )
        await self._session.flush()

    async def get(self, chunk_id: ChunkId) -> Optional[Chunk]:
        """Load one canonical chunk by business UUID."""

        record = await self._session.scalar(
            select(ChunkRecord)
            .options(joinedload(ChunkRecord.document))
            .where(ChunkRecord.id == chunk_id)
        )
        if record is None:
            return None
        return map_chunk_to_domain(record)

    async def get_contexts(self, ids: Sequence[ChunkId]) -> Sequence[ChunkContext]:
        """Load chunk contexts in input order without mutating retrieval orchestration."""

        if not ids:
            return []

        result = await self._session.scalars(
            select(ChunkRecord)
            .options(joinedload(ChunkRecord.document))
            .where(ChunkRecord.id.in_(ids))
        )
        records_by_id = {record.id: record for record in result}
        return [
            map_chunk_context_from_record(records_by_id[chunk_id])
            for chunk_id in ids
            if chunk_id in records_by_id
        ]

    async def _load_document_pks(self, chunks: Sequence[Chunk]) -> dict[DocumentId, int]:
        """Resolve business document UUIDs to internal primary keys for chunk inserts."""

        document_ids = {chunk.document_id for chunk in chunks}
        rows = await self._session.execute(
            select(DocumentRecord.id, DocumentRecord.pk).where(DocumentRecord.id.in_(document_ids))
        )
        document_pks = {document_id: document_pk for document_id, document_pk in rows}
        missing_document_ids = [
            document_id for document_id in document_ids if document_id not in document_pks
        ]
        if missing_document_ids:
            missing_preview = ", ".join(
                str(document_id) for document_id in missing_document_ids[:3]
            )
            raise ValueError(f"Cannot persist chunks without parent documents: {missing_preview}")
        return document_pks
