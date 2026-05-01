"""Mapping helpers for the TEI reranking HTTP API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict, cast

from rust_assistant.application.ports.reranking_client import (
    RerankingCandidate,
    RerankingResult,
)


class TeiRerankingRequest(TypedDict):
    """Request body accepted by the TEI /rerank endpoint."""

    query: str
    raw_scores: bool
    return_text: bool
    texts: list[str]
    truncate: bool
    truncation_direction: str


def map_reranking_request(
    query: str,
    candidates: Sequence[RerankingCandidate],
) -> TeiRerankingRequest:
    """Map application reranking input into a TEI /rerank request body."""
    return {
        "query": query,
        "raw_scores": False,
        "return_text": False,
        "texts": [candidate.text for candidate in candidates],
        "truncate": False,
        "truncation_direction": "right",
    }


def map_reranking_response(
    payload: object,
    candidates: Sequence[RerankingCandidate],
) -> list[RerankingResult]:
    """Map a TEI /rerank response body into application reranking results."""
    if not isinstance(payload, list):
        raise ValueError("TEI reranking response must be a list")

    items = cast(list[object], payload)
    seen_indexes: set[int] = set()
    results: list[RerankingResult] = []
    for item in items:
        index = _read_index(item, candidate_count=len(candidates))
        if index in seen_indexes:
            raise ValueError(f"TEI reranking response contains duplicate index: {index}")
        seen_indexes.add(index)
        results.append(
            RerankingResult(
                chunk_id=candidates[index].chunk_id,
                score=_read_score(item),
            )
        )
    return results


def _read_index(item: object, *, candidate_count: int) -> int:
    if not isinstance(item, Mapping):
        raise ValueError("TEI reranking item must be an object")

    mapping = cast(Mapping[str, object], item)
    index = mapping.get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("TEI reranking item index must be an integer")
    if index < 0 or index >= candidate_count:
        raise ValueError(f"TEI reranking item index out of range: {index}")
    return index


def _read_score(item: object) -> float:
    if not isinstance(item, Mapping):
        raise ValueError("TEI reranking item must be an object")

    mapping = cast(Mapping[str, object], item)
    score = mapping.get("score")
    if not isinstance(score, (float, int)) or isinstance(score, bool):
        raise ValueError("TEI reranking item score must be a number")
    return float(score)
