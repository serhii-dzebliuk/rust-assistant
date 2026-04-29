import json

import httpx
import pytest

from rust_assistant.application.ports.embedding_client import EmbeddingInput
from rust_assistant.infrastructure.adapters.embedding.tei.tei_embedding_client import (
    TeiEmbeddingClient,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_embed_text_unwraps_single_tei_embedding_response():
    async def handler(request):
        payload = json.loads(request.read())
        assert payload["inputs"] == "async"
        return httpx.Response(200, json=[[0.1, 0.2, 0.3]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        vector = await TeiEmbeddingClient(
            client=client,
            base_url="http://tei",
        ).embed_text("async")

    assert vector == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_texts_logs_embedding_progress(caplog):
    async def handler(request):
        inputs = json.loads(request.read())["inputs"]
        return httpx.Response(200, json=[[0.1] for _ in inputs])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with caplog.at_level("INFO"):
            await TeiEmbeddingClient(
                client=client,
                base_url="http://tei",
                max_batch_items=2,
                max_batch_tokens=100,
            ).embed_texts(
                [
                    EmbeddingInput(text="a", token_count=1),
                    EmbeddingInput(text="b", token_count=1),
                    EmbeddingInput(text="c", token_count=1),
                ]
            )

    assert "Embedding chunks: 0/3 (0.0%)" in caplog.messages
    assert "Embedding chunks: 2/3 (66.7%)" in caplog.messages
    assert "Embedding chunks: 3/3 (100.0%)" in caplog.messages


@pytest.mark.asyncio
async def test_embed_texts_returns_without_progress_logs_for_empty_input(caplog):
    async def handler(_request):
        raise AssertionError("No request expected")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with caplog.at_level("INFO"):
            vectors = await TeiEmbeddingClient(
                client=client,
                base_url="http://tei",
            ).embed_texts([])

    assert vectors == []
    assert caplog.messages == []


@pytest.mark.asyncio
async def test_embed_texts_includes_tei_response_body_in_http_error():
    async def handler(_request):
        return httpx.Response(429, json={"error": "Model is overloaded"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError, match="Model is overloaded"):
            await TeiEmbeddingClient(
                client=client,
                base_url="http://tei",
            ).embed_texts([EmbeddingInput(text="a", token_count=1)])
