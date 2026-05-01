"""Port for reranking retrieved text candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from rust_assistant.domain.value_objects.identifiers import ChunkId


@dataclass(frozen=True)
class RerankingCandidate:
    """One retrieved text candidate submitted for reranking."""

    chunk_id: ChunkId
    text: str


@dataclass(frozen=True)
class RerankingResult:
    """One reranked candidate score returned by a reranking provider."""

    chunk_id: ChunkId
    score: float


class RerankingClient(Protocol):
    """Rerank retrieved text candidates for a query."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankingCandidate],
    ) -> list[RerankingResult]:
        """Return provider-ranked candidates for the query."""
        ...
