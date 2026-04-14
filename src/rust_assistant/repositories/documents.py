"""Repositories for canonical document persistence."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rust_assistant.ingest.entities import Document
from rust_assistant.models import DocumentRecord


class DocumentRepository:
    """Persist and query canonical documents."""

    async def upsert_documents(
        self,
        session: AsyncSession,
        documents: Sequence[Document],
    ) -> dict[str, DocumentRecord]:
        """Insert or update documents keyed by their unique source path."""
        if not documents:
            return {}

        source_paths = [document.source_path for document in documents]
        existing_result = await session.scalars(
            select(DocumentRecord).where(DocumentRecord.source_path.in_(source_paths))
        )
        existing = {record.source_path: record for record in existing_result}

        for document in documents:
            metadata = document.metadata
            if not metadata.url:
                raise ValueError(f"Document URL is required for source_path={document.source_path}")

            record = existing.get(document.source_path)
            if record is None:
                record = DocumentRecord(source_path=document.source_path)
                session.add(record)
                existing[document.source_path] = record

            record.crate = metadata.crate.value
            record.title = document.title
            record.text_content = document.text
            record.parsed_content = [
                block.model_dump(mode="json") for block in document.structured_blocks
            ]
            record.url = metadata.url
            record.item_path = metadata.item_path
            record.rust_version = metadata.rust_version
            record.item_type = metadata.item_type.value if metadata.item_type else None

        await session.flush()
        return existing


__all__ = ["DocumentRepository"]
