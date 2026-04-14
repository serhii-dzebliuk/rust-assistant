"""Repositories for canonical chunk persistence and lookup."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from rust_assistant.ingest.entities import Chunk
from rust_assistant.models import ChunkRecord, DocumentRecord


class ChunkRepository:
    """Persist and query canonical chunks."""

    async def upsert_chunks(
        self,
        session: AsyncSession,
        chunks: Sequence[Chunk],
        documents_by_source_path: dict[str, DocumentRecord],
    ) -> list[ChunkRecord]:
        """Insert or update chunks keyed by `(document_id, chunk_index)`."""
        if not chunks:
            return []

        document_ids = {
            documents_by_source_path[chunk.metadata.doc_source_path].id for chunk in chunks
        }
        chunk_indexes = [chunk.metadata.chunk_index for chunk in chunks]
        existing_result = await session.scalars(
            select(ChunkRecord).where(
                ChunkRecord.document_id.in_(document_ids),
                ChunkRecord.chunk_index.in_(chunk_indexes),
            )
        )
        existing = {
            (record.document_id, record.chunk_index): record for record in existing_result
        }

        ordered_records: list[ChunkRecord] = []
        for chunk in chunks:
            document_record = documents_by_source_path[chunk.metadata.doc_source_path]
            metadata = chunk.metadata
            key = (document_record.id, metadata.chunk_index)
            record = existing.get(key)
            if record is None:
                record = ChunkRecord(
                    document_id=document_record.id,
                    chunk_index=metadata.chunk_index,
                )
                session.add(record)
                existing[key] = record

            record.text = chunk.text
            record.hash = chunk.text_hash or Chunk.compute_text_hash(chunk.text)
            record.token_count = None
            record.section_title = metadata.section
            record.section_anchor = metadata.anchor
            record.section_path = metadata.section_path
            record.start_offset = metadata.start_char
            record.end_offset = metadata.end_char
            ordered_records.append(record)

        await session.flush()
        return ordered_records

    async def list_by_chunk_ids(
        self,
        session: AsyncSession,
        chunk_ids: Sequence[int],
    ) -> list[ChunkRecord]:
        """Load chunks and their parent documents in the same order as the requested ids."""
        if not chunk_ids:
            return []

        result = await session.scalars(
            select(ChunkRecord)
            .options(joinedload(ChunkRecord.document))
            .where(ChunkRecord.id.in_(chunk_ids))
        )
        records_by_id = {record.id: record for record in result}
        return [records_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in records_by_id]


__all__ = ["ChunkRepository"]
