"""Mapping helpers for search API transport models."""

from __future__ import annotations

from rust_assistant.application.use_cases.search import (
    SearchResult,
    SearchResultHit,
)
from rust_assistant.infrastructure.entrypoints.api.schemas.search import (
    SearchHit,
    SearchResponse,
)


def map_search_result_to_response(result: SearchResult) -> SearchResponse:
    """Map an application search result into the HTTP response schema."""
    hits = [_map_hit(hit) for hit in result.hits]
    return SearchResponse(
        query=result.query,
        total_results=len(hits),
        results=hits,
    )


def _map_hit(hit: SearchResultHit) -> SearchHit:
    return SearchHit(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        title=hit.title,
        source_path=hit.source_path,
        url=hit.url,
        section=hit.section,
        item_path=hit.item_path,
        crate=hit.crate,
        item_type=hit.item_type,
        rust_version=hit.rust_version,
        score=hit.score,
        text=hit.text,
    )
