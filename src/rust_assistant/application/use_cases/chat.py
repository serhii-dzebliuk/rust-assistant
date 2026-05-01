"""Chat use-case orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from rust_assistant.application.services.retrieval.models import (
    RetrievalRequest,
    RetrievedChunk,
)
from rust_assistant.application.services.retrieval.pipeline import RetrievalPipeline


@dataclass(slots=True, frozen=True)
class ChatCommand:
    """Input command for the chat use case."""

    question: str
    retrieval_limit: int = 20
    reranking_limit: int = 5
    use_reranking: bool = True


@dataclass(slots=True, frozen=True)
class ChatResult:
    """Chat use-case result."""

    answer: str
    sources: list[ChatResultSource]


@dataclass(slots=True, frozen=True)
class ChatResultSource:
    """One retrieved source used by the chat use case."""

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


class ChatUseCase:
    """Retrieve grounded sources for a future generated chat answer."""

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipeline,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline

    async def execute(self, command: ChatCommand) -> ChatResult:
        """Retrieve sources for a chat question."""
        question = command.question.strip()
        chunks = await self._retrieval_pipeline.retrieve(
            RetrievalRequest(
                query=question,
                retrieval_limit=command.retrieval_limit,
                reranking_limit=command.reranking_limit,
                use_reranking=command.use_reranking,
            )
        )
        # TODO: Generate a grounded answer through an LLM after retrieval is stable.
        return ChatResult(
            answer="",
            sources=[_map_source(chunk) for chunk in chunks],
        )


def _map_source(chunk: RetrievedChunk) -> ChatResultSource:
    return ChatResultSource(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        source_path=chunk.source_path,
        url=chunk.url,
        section=chunk.section,
        item_path=chunk.item_path,
        crate=chunk.crate,
        item_type=chunk.item_type,
        rust_version=chunk.rust_version,
        score=chunk.score,
        text=chunk.text,
    )
