"""Vector store abstraction for retrieval and ingest synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(slots=True, frozen=True)
class VectorSearchHit:
    """Single vector-search hit returned by the vector store."""

    chunk_id: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VectorChunkPayload:
    """Chunk payload written to the vector store during ingest."""

    chunk_id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStoreClient(Protocol):
    """Interface for vector search and indexing backends."""

    async def search(
        self,
        *,
        query: str,
        k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> list[VectorSearchHit]:
        """Return the most relevant vector hits for a query."""

    async def upsert_chunks(self, *, chunks: Sequence[VectorChunkPayload]) -> None:
        """Insert or update chunk vectors in the backing vector store."""

    async def ping(self) -> bool:
        """Check whether the vector store is reachable."""


class StubVectorStoreClient:
    """No-op vector store used until a real Qdrant adapter is connected."""

    async def search(
        self,
        *,
        query: str,
        k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> list[VectorSearchHit]:
        """Return no hits while the real vector store is not connected."""
        _ = query, k, filters
        return []

    async def upsert_chunks(self, *, chunks: Sequence[VectorChunkPayload]) -> None:
        """Accept chunk writes without performing real indexing."""
        _ = chunks

    async def ping(self) -> bool:
        """Report that the stub vector store is available for scaffolding."""
        return True


__all__ = [
    "StubVectorStoreClient",
    "VectorChunkPayload",
    "VectorSearchHit",
    "VectorStoreClient",
]
