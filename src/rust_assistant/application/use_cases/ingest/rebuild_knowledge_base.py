"""Use case for rebuilding the canonical knowledge base from ingest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.ports.chunk_token_counter import ChunkTokenCounterPort
from rust_assistant.application.ports.uow import UnitOfWork


@dataclass(slots=True, frozen=True)
class RebuildKnowledgeBaseResult:
    """Summary of a completed knowledge-base rebuild."""

    status: str
    document_count: int
    chunk_count: int
    deleted_document_count: int = 0
    deleted_chunk_count: int = 0


class RebuildKnowledgeBase:
    """Rebuild all canonical documents and chunks from completed ingest artifacts."""

    def _validate_artifact_integrity(self, artifacts: IngestPipelineArtifacts) -> None:
        """Validate document/chunk consistency before rebuilding the knowledge base."""
        if not artifacts.deduped_docs:
            raise ValueError("Refusing to rebuild Postgres ingest data with zero documents")
        if not artifacts.deduped_chunks:
            raise ValueError("Refusing to rebuild Postgres ingest data with zero chunks")

        document_source_paths = {document.source_path for document in artifacts.deduped_docs}
        chunk_source_paths = {chunk.source_path for chunk in artifacts.deduped_chunks}

        documents_without_chunks = sorted(document_source_paths - chunk_source_paths)
        if documents_without_chunks:
            preview = ", ".join(documents_without_chunks[:5])
            raise ValueError(
                f"Refusing knowledge-base rebuild with documents without chunks: {preview}"
            )

        chunks_without_documents = sorted(chunk_source_paths - document_source_paths)
        if chunks_without_documents:
            preview = ", ".join(chunks_without_documents[:5])
            raise ValueError(
                "Refusing knowledge-base rebuild with chunks without matching documents: "
                f"{preview}"
            )

    async def execute(
        self,
        *,
        artifacts: IngestPipelineArtifacts,
        uow: UnitOfWork,
        token_counter: Optional[ChunkTokenCounterPort] = None,
    ) -> RebuildKnowledgeBaseResult:
        """Replace the whole persisted knowledge base with the provided ingest artifacts."""
        self._validate_artifact_integrity(artifacts)
        document_count = len(artifacts.deduped_docs)
        chunks_to_persist = artifacts.deduped_chunks
        if token_counter is not None:
            chunks_to_persist = token_counter.with_token_counts(chunks_to_persist)

        async with uow:
            await uow.documents.delete_all()
            await uow.documents.add_many(artifacts.deduped_docs)
            await uow.chunks.add_many(chunks_to_persist)
            await uow.commit()

        return RebuildKnowledgeBaseResult(
            status="completed",
            document_count=document_count,
            chunk_count=len(chunks_to_persist),
        )
