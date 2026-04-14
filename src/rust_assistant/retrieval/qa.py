"""QA orchestration for the serving runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from rust_assistant.clients.llm import LLMClient, StubLLMClient

from .prompt import PromptBuilder
from .retriever import RetrievedChunk, Retriever, StubRetriever


@dataclass(slots=True, frozen=True)
class QADebugData:
    """Debug details collected while producing a chat answer."""

    retrieval_time_ms: float
    model_name: str
    retrieved_sources: int


@dataclass(slots=True, frozen=True)
class QAResult:
    """Result of the runtime QA pipeline."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    confidence: str = "unknown"
    debug_data: Optional[QADebugData] = None


class QAPipeline:
    """Coordinate retrieval, prompt building, and generation for chat."""

    def __init__(
        self,
        *,
        retriever: Optional[Retriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self._retriever = retriever or StubRetriever()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._llm = llm or StubLLMClient()

    async def answer(
        self,
        *,
        question: str,
        k: int,
        filters: Optional[Mapping[str, Any]] = None,
        debug: bool = False,
        session: Optional[AsyncSession] = None,
    ) -> QAResult:
        """Generate a chat answer using the runtime RAG building blocks."""
        retrieval = await self._retriever.search(
            query=question,
            k=k,
            filters=filters,
            session=session,
        )
        prompt = self._prompt_builder.build_chat_prompt(
            question=question,
            context_chunks=retrieval.hits,
        )
        llm_result = await self._llm.answer(
            prompt=prompt,
            question=question,
            context_chunks=retrieval.hits,
        )

        debug_data = None
        if debug:
            debug_data = QADebugData(
                retrieval_time_ms=retrieval.retrieval_time_ms,
                model_name=llm_result.model_name,
                retrieved_sources=len(retrieval.hits),
            )

        return QAResult(
            answer=llm_result.answer,
            sources=list(retrieval.hits),
            confidence=llm_result.confidence,
            debug_data=debug_data,
        )
