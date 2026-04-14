"""Persistence orchestration for ingest outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rust_assistant.core.db import AsyncSessionFactory
from rust_assistant.ingest.pipeline import PipelineArtifacts
from rust_assistant.repositories import ChunkRepository, DocumentRepository


@dataclass(slots=True, frozen=True)
class IngestPersistenceResult:
    """Summary of a persisted ingest execution."""

    status: str
    document_count: int
    chunk_count: int


async def persist_ingest_artifacts(
    *,
    artifacts: PipelineArtifacts,
    session_factory: Optional[AsyncSessionFactory],
) -> IngestPersistenceResult:
    """Persist ingest artifacts to Postgres."""
    if session_factory is None:
        raise ValueError("DATABASE_URL must be configured before persisting ingest artifacts")

    document_repository = DocumentRepository()
    chunk_repository = ChunkRepository()

    document_count = len(artifacts.deduped_docs)
    chunk_count = len(artifacts.deduped_chunks)

    async with session_factory() as session:
        documents_by_source_path = await document_repository.upsert_documents(
            session,
            artifacts.deduped_docs,
        )
        await chunk_repository.upsert_chunks(
            session,
            artifacts.deduped_chunks,
            documents_by_source_path,
        )
        await session.commit()

    return IngestPersistenceResult(
        status="completed",
        document_count=document_count,
        chunk_count=chunk_count,
    )
