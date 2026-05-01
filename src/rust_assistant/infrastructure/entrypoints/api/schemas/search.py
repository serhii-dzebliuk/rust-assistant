"""Search endpoint schemas."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=1000)
    retrieval_limit: int = Field(default=50, ge=1, le=100)
    reranking_limit: int = Field(default=10, ge=1, le=100)
    use_reranking: bool = True

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Reject empty or whitespace-only query values."""
        if not value or not value.strip():
            raise ValueError("Query cannot be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_limit_order(self) -> "SearchRequest":
        """Reject final reranking limits above the retrieval pool size."""
        if self.reranking_limit > self.retrieval_limit:
            raise ValueError("reranking_limit must be <= retrieval_limit")
        return self


class SearchHit(BaseModel):
    """One hydrated search hit returned to the API client."""

    chunk_id: UUID
    document_id: UUID
    title: str
    source_path: str
    url: str
    section: Optional[str] = None
    item_path: Optional[str] = None
    crate: Optional[str] = None
    item_type: Optional[str] = None
    rust_version: Optional[str] = None
    score: float
    text: str


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    query: str
    total_results: int
    results: list[SearchHit] = Field(default_factory=list[SearchHit])
