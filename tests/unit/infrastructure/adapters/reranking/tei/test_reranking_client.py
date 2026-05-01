import json
from uuid import uuid4

import httpx
import pytest

from rust_assistant.application.ports.reranking_client import RerankingCandidate
from rust_assistant.domain.value_objects.identifiers import ChunkId
from rust_assistant.infrastructure.adapters.reranking.tei.tei_reranking_client import (
    TeiRerankingClient,
)

pytestmark = pytest.mark.unit


def _candidate(text: str) -> RerankingCandidate:
    return RerankingCandidate(chunk_id=ChunkId(uuid4()), text=text)


@pytest.mark.asyncio
async def test_rerank_posts_tei_request_and_maps_response():
    first = _candidate("first")
    second = _candidate("second")

    async def handler(request):
        assert request.url == "http://tei/rerank"
        payload = json.loads(request.read())
        assert payload == {
            "query": "async",
            "raw_scores": False,
            "return_text": False,
            "texts": ["first", "second"],
            "truncate": False,
            "truncation_direction": "right",
        }
        return httpx.Response(
            200,
            json=[
                {"index": 1, "score": 0.99},
                {"index": 0, "score": 0.75},
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await TeiRerankingClient(client=client, base_url="http://tei").rerank(
            "async",
            [first, second],
        )

    assert [result.chunk_id for result in results] == [second.chunk_id, first.chunk_id]
    assert [result.score for result in results] == [0.99, 0.75]


@pytest.mark.asyncio
async def test_rerank_returns_without_request_for_empty_candidates():
    async def handler(_request):
        raise AssertionError("No request expected")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await TeiRerankingClient(client=client, base_url="http://tei").rerank(
            "async",
            [],
        )

    assert results == []


@pytest.mark.asyncio
async def test_rerank_includes_tei_response_body_in_http_error():
    async def handler(_request):
        return httpx.Response(429, json={"error": "Model is overloaded"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError, match="Model is overloaded"):
            await TeiRerankingClient(client=client, base_url="http://tei").rerank(
                "async",
                [_candidate("first")],
            )
