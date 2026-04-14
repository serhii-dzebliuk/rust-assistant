"""Application service for chat use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from rust_assistant.retrieval.qa import QAPipeline
from rust_assistant.retrieval.retriever import RetrievedChunk


@dataclass(slots=True, frozen=True)
class ChatDebugData:
    """Optional debug payload produced by the chat service."""

    mode: str
    dependencies: dict[str, str] = field(default_factory=dict)
    retrieval_time_ms: Optional[float] = None
    model_name: Optional[str] = None
    retrieved_sources: Optional[int] = None


@dataclass(slots=True, frozen=True)
class ChatResult:
    """Chat response produced by the application service."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    confidence: str = "unknown"
    debug_info: Optional[ChatDebugData] = None
    mode: str = "stub"


class ChatService:
    """Application-layer wrapper around the runtime QA pipeline."""

    def __init__(
        self,
        *,
        mode: str = "stub",
        dependencies: Optional[Mapping[str, str]] = None,
        qa_pipeline: Optional[QAPipeline] = None,
        session: Optional[AsyncSession] = None,
    ) -> None:
        self._mode = mode
        self._dependencies = dict(dependencies or {})
        self._qa_pipeline = qa_pipeline or QAPipeline()
        self._session = session

    async def chat(
        self,
        *,
        question: str,
        k: int,
        filters: Optional[Mapping[str, Any]] = None,
        debug: bool = False,
    ) -> ChatResult:
        """Execute the runtime QA flow and adapt it to the service response shape."""
        qa_result = await self._qa_pipeline.answer(
            question=question,
            k=k,
            filters=filters,
            debug=debug,
            session=self._session,
        )

        debug_info = None
        if debug:
            debug_info = ChatDebugData(
                mode=self._mode,
                dependencies=dict(self._dependencies),
                retrieval_time_ms=(
                    qa_result.debug_data.retrieval_time_ms
                    if qa_result.debug_data is not None
                    else None
                ),
                model_name=(
                    qa_result.debug_data.model_name
                    if qa_result.debug_data is not None
                    else None
                ),
                retrieved_sources=(
                    qa_result.debug_data.retrieved_sources
                    if qa_result.debug_data is not None
                    else None
                ),
            )

        return ChatResult(
            question=question,
            answer=qa_result.answer,
            sources=list(qa_result.sources),
            confidence=qa_result.confidence,
            debug_info=debug_info,
            mode=self._mode,
        )
