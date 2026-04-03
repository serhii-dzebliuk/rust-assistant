"""Search endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from rustrag.core.models import Crate, ItemType


class SearchFilters(BaseModel):
    """Optional filters accepted by the search endpoint."""

    crate: Crate | None = None
    item_type: ItemType | None = None


class SearchHit(BaseModel):
    """One search hit returned to the API client."""

    title: str
    source_path: str
    section: str | None = None
    item_path: str | None = None
    score: float = Field(..., ge=0.0, le=1.0)
    snippet: str


class SearchRequest(BaseModel):
    """Request body for POST /api/search."""

    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=50)
    filters: SearchFilters | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Reject empty or whitespace-only query values."""
        if not value or not value.strip():
            raise ValueError("Query cannot be empty")
        return value.strip()


class SearchResponse(BaseModel):
    """Response body for POST /api/search."""

    query: str
    total_results: int
    results: list[SearchHit] = Field(default_factory=list)
    # TODO: Populate latency from the real retrieval pipeline instead of the
    # stub zero value.
    retrieval_time_ms: float = 0.0
    # TODO: Remove the stub default once the route always returns runtime mode.
    mode: str = "stub"

