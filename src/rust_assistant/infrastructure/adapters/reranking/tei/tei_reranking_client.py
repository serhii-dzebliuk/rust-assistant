"""TEI reranking adapter implementation."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from rust_assistant.application.ports.reranking_client import (
    RerankingCandidate,
    RerankingResult,
)
from rust_assistant.infrastructure.adapters.reranking.tei.mappers import (
    map_reranking_request,
    map_reranking_response,
)


class TeiRerankingClient:
    """Rerank text candidates through Hugging Face Text Embeddings Inference."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RerankingCandidate],
    ) -> list[RerankingResult]:
        """Rerank candidate texts for one query."""
        if not candidates:
            return []

        response = await self._client.post(
            f"{self._base_url}/rerank",
            json=map_reranking_request(query, candidates),
        )
        self._raise_for_status(response, candidate_count=len(candidates))
        return map_reranking_response(response.json(), candidates)

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        candidate_count: int,
    ) -> None:
        """Raise an HTTPStatusError with TEI response details included."""
        if response.is_success:
            return

        message = (
            f"TEI reranking request failed with status={response.status_code} "
            f"candidate_count={candidate_count} body={response.text!r}"
        )
        raise httpx.HTTPStatusError(
            message,
            request=response.request,
            response=response,
        )
