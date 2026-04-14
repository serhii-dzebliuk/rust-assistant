"""Search endpoints for the serving API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rust_assistant.retrieval.retriever import RetrievedChunk
from rust_assistant.schemas.search import SearchHit, SearchRequest, SearchResponse
from rust_assistant.services.search_service import SearchService

from ..deps import get_search_service


router = APIRouter(tags=["search"])


def _to_search_hit(hit: RetrievedChunk) -> SearchHit:
    """Convert a service-level search hit into the HTTP schema."""
    return SearchHit(
        title=hit.title,
        source_path=hit.source_path,
        section=hit.section,
        item_path=hit.item_path,
        crate=hit.crate,
        item_type=hit.item_type,
        score=hit.score,
        snippet=hit.snippet,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search endpoint backed by the search application service."""
    search_result = await search_service.search(
        query=payload.query,
        k=payload.k,
        filters=payload.filters.model_dump(mode="json", exclude_none=True)
        if payload.filters
        else None,
    )

    return SearchResponse(
        query=search_result.query,
        total_results=search_result.total_results,
        results=[_to_search_hit(hit) for hit in search_result.results],
        retrieval_time_ms=search_result.retrieval_time_ms,
        mode=search_result.mode,
    )
