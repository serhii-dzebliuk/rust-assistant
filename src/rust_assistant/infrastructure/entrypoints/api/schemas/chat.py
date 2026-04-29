"""Chat endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .search import SearchHit


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Reject empty or whitespace-only questions."""
        if not value or not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    answer: str
    sources: list[SearchHit] = Field(default_factory=list)
