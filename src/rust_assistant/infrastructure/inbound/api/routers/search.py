"""Search endpoints for the serving API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from rust_assistant.infrastructure.inbound.api.schemas.search import SearchRequest, SearchResponse


router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    """Search endpoint placeholder until the real implementation is built."""
    _ = payload
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Not implemented",
    )
