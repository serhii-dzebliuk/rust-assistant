"""Chat endpoints for the serving API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from rust_assistant.infrastructure.inbound.api.schemas.chat import ChatRequest, ChatResponse


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """Chat endpoint placeholder until the real implementation is built."""
    _ = payload
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Not implemented",
    )
