"""Retrieval search orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from rust_assistant.application.dto.chunk_context import ChunkContext
from rust_assistant.application.ports.embedding_client import EmbeddingClient
from rust_assistant.application.ports.reranking_client import (
    RerankingCandidate,
    RerankingClient,
    RerankingResult,
)
from rust_assistant.application.ports.uow import UnitOfWork
from rust_assistant.application.ports.vector_storage import VectorStorage
from rust_assistant.domain.value_objects.identifiers import ChunkId


@dataclass(slots=True, frozen=True)
class SearchCommand:
    """Input command for the search use case."""

    query: str
    retrieval_limit: int = 50
    reranking_limit: int = 10


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Search use-case result."""

    query: str
    hits: list[SearchResultHit]


@dataclass(slots=True, frozen=True)
class SearchResultHit:
    """One hydrated search hit produced by the application layer."""

    chunk_id: UUID
    document_id: UUID
    title: str
    source_path: str
    url: str
    section: Optional[str]
    item_path: Optional[str]
    crate: Optional[str]
    item_type: Optional[str]
    rust_version: Optional[str]
    score: float
    text: str


class SearchUseCase:
    """Retrieve and hydrate documentation chunks for a user query."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        vector_storage: VectorStorage,
        reranking_client: RerankingClient,
        uow: UnitOfWork,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_storage = vector_storage
        self._reranking_client = reranking_client
        self._uow = uow

    async def execute(self, command: SearchCommand) -> SearchResult:
        """Run vector retrieval and load canonical chunk contexts."""
        query = command.query.strip()
        query_vector = await self._embedding_client.embed_text(query)
        vector_hits = await self._vector_storage.search(
            query_vector=query_vector,
            limit=command.retrieval_limit,
        )
        if not vector_hits:
            return SearchResult(query=query, hits=[])

        chunk_ids = [ChunkId(hit.chunk_id) for hit in vector_hits]
        async with self._uow as uow:
            contexts = await uow.chunks.get_contexts(chunk_ids)

        contexts_by_id = {context.chunk_id: context for context in contexts}
        reranking_candidates = [
            RerankingCandidate(
                chunk_id=ChunkId(hit.chunk_id),
                text=contexts_by_id[ChunkId(hit.chunk_id)].text,
            )
            for hit in vector_hits
            if ChunkId(hit.chunk_id) in contexts_by_id
        ]
        if not reranking_candidates:
            return SearchResult(query=query, hits=[])

        reranking_results = await self._reranking_client.rerank(
            query=query,
            candidates=reranking_candidates,
        )
        selected_results = reranking_results[: command.reranking_limit]

        return SearchResult(
            query=query,
            hits=[
                _build_result_hit(result, contexts_by_id[result.chunk_id])
                for result in selected_results
                if result.chunk_id in contexts_by_id
            ],
        )


def _build_result_hit(result: RerankingResult, context: ChunkContext) -> SearchResultHit:
    """Combine reranking data with canonical chunk context."""
    return SearchResultHit(
        chunk_id=context.chunk_id,
        document_id=context.document_id,
        title=context.title,
        source_path=context.source_path,
        url=context.url,
        section=context.section_title,
        item_path=context.item_path,
        crate=context.crate.value,
        item_type=context.item_type.value if context.item_type is not None else None,
        rust_version=context.rust_version,
        score=result.score,
        text=context.text,
    )
