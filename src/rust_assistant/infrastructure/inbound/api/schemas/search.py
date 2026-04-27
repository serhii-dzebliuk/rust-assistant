"""Search endpoint schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from rust_assistant.application.dto.search import SearchFilters, SearchHit


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=50)
    filters: Optional[SearchFilters] = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Reject empty or whitespace-only query values."""
        if not value or not value.strip():
            raise ValueError("Query cannot be empty")
        return value.strip()


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    query: str
    total_results: int
    results: list[SearchHit] = Field(default_factory=list[SearchHit])
