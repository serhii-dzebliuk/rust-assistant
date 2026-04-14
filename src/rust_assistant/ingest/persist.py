"""Persistence orchestration for ingest outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rust_assistant.clients.vectordb import (
    StubVectorStoreClient,
    VectorChunkPayload,
    VectorStoreClient,
)
from rust_assistant.core.db import AsyncSessionFactory
from rust_assistant.ingest.pipeline import PipelineArtifacts
from rust_assistant.repositories import ChunkRepository, DocumentRepository


@dataclass(slots=True, frozen=True)
class IngestPersistenceResult:
    """Summary of a persisted ingest execution."""

    status: str
    document_count: int
    chunk_count: int
    synced_chunk_count: int
    failed_chunk_count: int


async def persist_ingest_artifacts(
    *,
    artifacts: PipelineArtifacts,
    session_factory: Optional[AsyncSessionFactory],
    vector_store: Optional[VectorStoreClient] = None,
) -> IngestPersistenceResult:
    """Persist ingest artifacts to Postgres and synchronize them to the vector store."""
    if session_factory is None:
        raise ValueError("DATABASE_URL must be configured before persisting ingest artifacts")

    vector_store_client = vector_store or StubVectorStoreClient()
    document_repository = DocumentRepository()
    chunk_repository = ChunkRepository()

    document_count = len(artifacts.deduped_docs)
    chunk_count = len(artifacts.deduped_chunks)

    async with session_factory() as session:
        documents_by_source_path = await document_repository.upsert_documents(
            session,
            artifacts.deduped_docs,
        )
        chunk_records = await chunk_repository.upsert_chunks(
            session,
            artifacts.deduped_chunks,
            documents_by_source_path,
        )
        await session.commit()

    vector_payloads = [
        VectorChunkPayload(
            chunk_id=chunk_record.id,
            text=chunk.text,
            metadata={
                "crate": chunk.metadata.crate.value,
                "item_path": chunk.metadata.item_path,
                "item_type": chunk.metadata.item_type.value if chunk.metadata.item_type else None,
                "section": chunk.metadata.section,
                "source_path": chunk.metadata.doc_source_path,
            },
        )
        for chunk, chunk_record in zip(artifacts.deduped_chunks, chunk_records)
    ]

    await vector_store_client.upsert_chunks(chunks=vector_payloads)

    return IngestPersistenceResult(
        status="completed",
        document_count=document_count,
        chunk_count=chunk_count,
        synced_chunk_count=chunk_count,
        failed_chunk_count=0,
    )
