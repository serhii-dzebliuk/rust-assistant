"""Shared retrieval pipeline for search and chat use cases."""

from __future__ import annotations

from rust_assistant.application.dto.chunk_context import ChunkContext
from rust_assistant.application.ports.embedding_client import EmbeddingClient
from rust_assistant.application.ports.reranking_client import (
    RerankingCandidate,
    RerankingClient,
    RerankingResult,
)
from rust_assistant.application.ports.uow import UnitOfWork
from rust_assistant.application.ports.vector_storage import VectorSearchHit, VectorStorage
from rust_assistant.application.services.retrieval.models import (
    RetrievalRequest,
    RetrievedChunk,
)
from rust_assistant.domain.value_objects.identifiers import ChunkId


class RetrievalPipeline:
    """Retrieve, hydrate, and optionally rerank documentation chunks."""

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

    async def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        """Run vector retrieval and load canonical chunk contexts."""
        query = request.query.strip()
        query_vector = await self._embedding_client.embed_text(query)
        vector_hits = await self._vector_storage.search(
            query_vector=query_vector,
            limit=request.retrieval_limit,
        )
        if not vector_hits:
            return []

        chunk_ids = [ChunkId(hit.chunk_id) for hit in vector_hits]
        async with self._uow as uow:
            contexts = await uow.chunks.get_contexts(chunk_ids)

        contexts_by_id = {context.chunk_id: context for context in contexts}
        if not request.use_reranking:
            return [
                _build_vector_chunk(hit, contexts_by_id[ChunkId(hit.chunk_id)])
                for hit in vector_hits[: request.reranking_limit]
                if ChunkId(hit.chunk_id) in contexts_by_id
            ]

        reranking_candidates = [
            RerankingCandidate(
                chunk_id=ChunkId(hit.chunk_id),
                text=contexts_by_id[ChunkId(hit.chunk_id)].text,
            )
            for hit in vector_hits
            if ChunkId(hit.chunk_id) in contexts_by_id
        ]
        if not reranking_candidates:
            return []

        reranking_results = await self._reranking_client.rerank(
            query=query,
            candidates=reranking_candidates,
        )
        selected_results = reranking_results[: request.reranking_limit]
        return [
            _build_reranked_chunk(result, contexts_by_id[result.chunk_id])
            for result in selected_results
            if result.chunk_id in contexts_by_id
        ]


def _build_vector_chunk(hit: VectorSearchHit, context: ChunkContext) -> RetrievedChunk:
    """Combine vector ranking data with canonical chunk context."""
    return RetrievedChunk(
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
        score=hit.score,
        text=context.text,
    )


def _build_reranked_chunk(result: RerankingResult, context: ChunkContext) -> RetrievedChunk:
    """Combine reranking data with canonical chunk context."""
    return RetrievedChunk(
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
