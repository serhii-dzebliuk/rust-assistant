"""Search endpoints for the serving API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from rust_assistant.application.use_cases.search import SearchCommand
from rust_assistant.infrastructure.entrypoints.api.mappers.search import (
    map_search_result_to_response,
)
from rust_assistant.infrastructure.entrypoints.api.schemas.search import (
    SearchRequest,
    SearchResponse,
)


router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, payload: SearchRequest) -> SearchResponse:
    """Retrieve relevant Rust documentation chunks for a query."""
    use_case = request.app.state.container.search_use_case
    if use_case is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search is not configured",
        )
    result = await use_case.execute(
        SearchCommand(
            query=payload.query,
            retrieval_limit=payload.retrieval_limit,
            reranking_limit=payload.reranking_limit,
            use_reranking=payload.use_reranking,
        )
    )
    return map_search_result_to_response(result)
