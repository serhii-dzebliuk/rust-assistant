"""Retrieval primitives for the serving runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from rust_assistant.clients.vectordb import VectorStoreClient
from rust_assistant.repositories import ChunkRepository


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """Normalized chunk returned by the runtime retriever."""

    title: str
    source_path: str
    section: Optional[str] = None
    item_path: Optional[str] = None
    crate: Optional[str] = None
    item_type: Optional[str] = None
    score: float = 0.0
    snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Retrieval output consumed by search and QA flows."""

    query: str
    hits: list[RetrievedChunk] = field(default_factory=list)
    total_results: int = 0
    retrieval_time_ms: float = 0.0


class Retriever(Protocol):
    """Interface for runtime retrieval implementations."""

    async def search(
        self,
        *,
        query: str,
        k: int,
        filters: Optional[Mapping[str, Any]] = None,
        session: Optional[AsyncSession] = None,
    ) -> RetrievalResult:
        """Return the best matching chunks for the incoming query."""


class StubRetriever:
    """No-op retriever used until Qdrant-backed retrieval is connected."""

    async def search(
        self,
        *,
        query: str,
        k: int,
        filters: Optional[Mapping[str, Any]] = None,
        session: Optional[AsyncSession] = None,
    ) -> RetrievalResult:
        """Return an empty result set while runtime storage is unavailable."""
        _ = k, filters, session
        return RetrievalResult(
            query=query,
            hits=[],
            total_results=0,
            retrieval_time_ms=0.0,
        )


class DatabaseBackedRetriever:
    """Retriever that combines vector search hits with canonical Postgres chunks."""

    def __init__(
        self,
        *,
        vector_store: VectorStoreClient,
        chunk_repository: Optional[ChunkRepository] = None,
    ) -> None:
        self._vector_store = vector_store
        self._chunk_repository = chunk_repository or ChunkRepository()

    async def search(
        self,
        *,
        query: str,
        k: int,
        filters: Optional[Mapping[str, Any]] = None,
        session: Optional[AsyncSession] = None,
    ) -> RetrievalResult:
        """Search the vector store and hydrate hits from canonical Postgres rows."""
        if session is None:
            return RetrievalResult(query=query)

        vector_hits = await self._vector_store.search(query=query, k=k, filters=filters)
        chunk_ids = [hit.chunk_id for hit in vector_hits]
        chunk_records = await self._chunk_repository.list_by_chunk_ids(session, chunk_ids)
        records_by_id = {record.id: record for record in chunk_records}

        hits: list[RetrievedChunk] = []
        for vector_hit in vector_hits:
            record = records_by_id.get(vector_hit.chunk_id)
            if record is None:
                continue
            hits.append(
                RetrievedChunk(
                    title=record.document.title,
                    source_path=record.document.source_path,
                    section=record.section_title,
                    item_path=record.document.item_path,
                    crate=record.document.crate,
                    item_type=record.document.item_type,
                    score=vector_hit.score,
                    snippet=_build_snippet(record.text),
                    metadata={
                        "chunk_id": record.id,
                    },
                )
            )

        return RetrievalResult(
            query=query,
            hits=hits,
            total_results=len(hits),
            retrieval_time_ms=0.0,
        )


def _build_snippet(text: str, max_length: int = 200) -> str:
    """Build a compact text preview for API responses."""
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."
