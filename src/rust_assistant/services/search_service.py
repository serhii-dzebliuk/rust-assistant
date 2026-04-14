"""Application service for search use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from rust_assistant.retrieval.retriever import RetrievedChunk, Retriever, StubRetriever


@dataclass(slots=True, frozen=True)
class SearchResultPage:
    """Search response produced by the application service."""

    query: str
    total_results: int
    results: list[RetrievedChunk] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    mode: str = "stub"


class SearchService:
    """Application-layer wrapper around the runtime retriever."""

    def __init__(
        self,
        *,
        mode: str = "stub",
        retriever: Optional[Retriever] = None,
        session: Optional[AsyncSession] = None,
    ) -> None:
        self._mode = mode
        self._retriever = retriever or StubRetriever()
        self._session = session

    async def search(
        self,
        *,
        query: str,
        k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> SearchResultPage:
        """Execute runtime retrieval and adapt it to the API-facing service shape."""
        retrieval = await self._retriever.search(
            query=query,
            k=k,
            filters=filters,
            session=self._session,
        )
        return SearchResultPage(
            query=retrieval.query,
            total_results=retrieval.total_results,
            results=list(retrieval.hits),
            retrieval_time_ms=retrieval.retrieval_time_ms,
            mode=self._mode,
        )
