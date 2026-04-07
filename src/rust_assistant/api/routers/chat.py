"""Chat endpoints for the serving API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rust_assistant.retrieval.retriever import RetrievedChunk
from rust_assistant.schemas.chat import ChatDebugInfo, ChatRequest, ChatResponse
from rust_assistant.schemas.search import SearchHit
from rust_assistant.services.chat_service import ChatService

from ..deps import get_chat_service


router = APIRouter(tags=["chat"])


def _to_search_hit(hit: RetrievedChunk) -> SearchHit:
    """Convert a service-level source item into the HTTP schema."""
    return SearchHit(
        title=hit.title,
        source_path=hit.source_path,
        section=hit.section,
        item_path=hit.item_path,
        score=hit.score,
        snippet=hit.snippet,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Chat endpoint backed by the chat application service."""
    chat_result = chat_service.chat(
        question=payload.question,
        k=payload.k,
        filters=payload.filters.model_dump(mode="json", exclude_none=True)
        if payload.filters
        else None,
        debug=payload.debug,
    )

    debug_info = None
    if chat_result.debug_info is not None:
        debug_info = ChatDebugInfo(
            mode=chat_result.debug_info.mode,
            dependencies=chat_result.debug_info.dependencies,
            retrieval_time_ms=chat_result.debug_info.retrieval_time_ms,
            model_name=chat_result.debug_info.model_name,
            retrieved_sources=chat_result.debug_info.retrieved_sources,
        )

    return ChatResponse(
        question=chat_result.question,
        answer=chat_result.answer,
        sources=[_to_search_hit(hit) for hit in chat_result.sources],
        confidence=chat_result.confidence,
        debug_info=debug_info,
        mode=chat_result.mode,
    )
