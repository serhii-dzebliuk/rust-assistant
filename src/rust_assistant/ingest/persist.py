"""Persistence orchestration for ingest outputs."""

from __future__ import annotations

from collections.abc import Sequence
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
    deleted_document_count: int = 0
    deleted_chunk_count: int = 0


def _validate_artifact_integrity(artifacts: PipelineArtifacts) -> None:
    """Validate document/chunk consistency before opening a DB transaction."""
    document_source_paths = {document.source_path for document in artifacts.deduped_docs}
    chunk_source_paths = {chunk.metadata.doc_source_path for chunk in artifacts.deduped_chunks}

    documents_without_chunks = sorted(document_source_paths - chunk_source_paths)
    if documents_without_chunks:
        preview = ", ".join(documents_without_chunks[:5])
        raise ValueError(f"Refusing to persist documents without chunks: {preview}")

    chunks_without_documents = sorted(chunk_source_paths - document_source_paths)
    if chunks_without_documents:
        preview = ", ".join(chunks_without_documents[:5])
        raise ValueError(f"Refusing to persist chunks without matching documents: {preview}")


async def persist_ingest_artifacts(
    *,
    artifacts: PipelineArtifacts,
    session_factory: Optional[AsyncSessionFactory],
    replace_crates: Sequence[str],
) -> IngestPersistenceResult:
    """Persist ingest artifacts to Postgres with atomic crate-scope replacement."""
    if session_factory is None:
        raise ValueError("DATABASE_URL must be configured before persisting ingest artifacts")
    if not replace_crates:
        raise ValueError("At least one crate must be selected for ingest replacement")
    if not artifacts.deduped_docs:
        raise ValueError("Refusing to replace Postgres ingest data with zero documents")
    if not artifacts.deduped_chunks:
        raise ValueError("Refusing to replace Postgres ingest data with zero chunks")
    _validate_artifact_integrity(artifacts)

    document_repository = DocumentRepository()
    chunk_repository = ChunkRepository()

    document_count = len(artifacts.deduped_docs)
    chunk_count = len(artifacts.deduped_chunks)
    deleted_document_count = 0
    deleted_chunk_count = 0

    async with session_factory() as session:
        async with session.begin():
            (
                deleted_document_count,
                deleted_chunk_count,
            ) = await document_repository.delete_by_crates(session, replace_crates)
            documents_by_source_path = await document_repository.upsert_documents(
                session,
                artifacts.deduped_docs,
            )
            await chunk_repository.upsert_chunks(
                session,
                artifacts.deduped_chunks,
                documents_by_source_path,
            )

    return IngestPersistenceResult(
        status="completed",
        document_count=document_count,
        chunk_count=chunk_count,
        deleted_document_count=deleted_document_count,
        deleted_chunk_count=deleted_chunk_count,
    )
