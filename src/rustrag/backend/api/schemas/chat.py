"""Chat endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .search import SearchFilters, SearchHit


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""

    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters | None = None
    debug: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Reject empty or whitespace-only questions."""
        if not value or not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()


class ChatDebugInfo(BaseModel):
    """Optional debug block returned by the chat endpoint."""

    mode: str = "stub"
    dependencies: dict[str, str] = Field(default_factory=dict)
    retrieval_time_ms: float | None = None
    model_name: str | None = None
    retrieved_sources: int | None = None


class ChatResponse(BaseModel):
    """Response body for POST /api/chat."""

    question: str
    answer: str
    sources: list[SearchHit] = Field(default_factory=list)
    confidence: str = "unknown"
    debug_info: ChatDebugInfo | None = None
    mode: str = "stub"
