"""Retrieval primitives for the serving runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """Normalized chunk returned by the runtime retriever."""

    title: str
    source_path: str
    section: str | None = None
    item_path: str | None = None
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

    def search(
        self,
        *,
        query: str,
        k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Return the best matching chunks for the incoming query."""


class StubRetriever:
    """No-op retriever used until Qdrant-backed retrieval is connected."""

    def search(
        self,
        *,
        query: str,
        k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Return an empty result set while runtime storage is unavailable."""
        _ = k, filters
        return RetrievalResult(
            query=query,
            hits=[],
            total_results=0,
            retrieval_time_ms=0.0,
        )
