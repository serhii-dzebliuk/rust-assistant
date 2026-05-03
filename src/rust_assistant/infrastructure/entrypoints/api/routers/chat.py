"""Chat endpoints for the serving API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from rust_assistant.application.use_cases.chat import ChatCommand, ChatQuestionTooLargeError
from rust_assistant.infrastructure.entrypoints.api.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Answer one independent chat question over retrieved Rust documentation."""
    use_case = getattr(request.app.state.container, "chat_use_case", None)
    if use_case is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat is not configured",
        )

    try:
        result = await use_case.execute(ChatCommand(question=payload.question))
    except ChatQuestionTooLargeError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Chat answer generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Chat generation failed",
        ) from exc

    return ChatResponse(answer=result.answer)
