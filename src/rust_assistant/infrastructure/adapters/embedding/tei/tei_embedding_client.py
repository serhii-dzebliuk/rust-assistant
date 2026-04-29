"""TEI embedding adapter implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any, Optional, Union

import httpx

from rust_assistant.application.ports.embedding_client import EmbeddingInput, EmbeddingVector

logger = logging.getLogger(__name__)


EmbedInput = Union[str, list[str]]


class TeiEmbeddingClient:
    """Generate embeddings through Hugging Face Text Embeddings Inference."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        normalize: bool = True,
        max_batch_tokens: int = 2048,
        max_batch_items: int = 64,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._normalize = normalize
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_items = max_batch_items

    async def embed_text(self, text: str) -> EmbeddingVector:
        """Embed one text input."""
        response = await self._post_embed(text)
        self._raise_for_status(response, batch_size=1, batch_tokens=None)
        return _parse_single_embedding(response.json())

    async def embed_texts(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingVector]:
        """Embed text inputs in token-aware batches."""
        total_inputs = len(inputs)
        vectors: list[EmbeddingVector] = []
        if not inputs:
            return vectors

        logger.info("Embedding chunks: 0/%s (0.0%%)", total_inputs)

        for batch in _iter_batches(
            inputs,
            max_batch_items=self._max_batch_items,
            max_batch_tokens=self._max_batch_tokens,
        ):
            batch_vectors = await self._embed_batch(batch)
            vectors.extend(batch_vectors)
            logger.info(
                "Embedding chunks: %s/%s (%.1f%%)",
                len(vectors),
                total_inputs,
                len(vectors) / total_inputs * 100,
            )

        return vectors

    async def _embed_batch(self, batch: Sequence[EmbeddingInput]) -> list[EmbeddingVector]:
        """Embed one configured batch."""
        response = await self._post_embed([item.text for item in batch])
        self._raise_for_status(
            response,
            batch_size=len(batch),
            batch_tokens=_sum_batch_tokens(batch),
        )
        return _parse_embedding_batch(response.json())

    async def _post_embed(self, inputs: EmbedInput) -> httpx.Response:
        """Send one raw TEI embed request."""
        return await self._client.post(
            f"{self._base_url}/embed",
            json={
                "inputs": inputs,
                "normalize": self._normalize,
            },
        )

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        batch_size: int,
        batch_tokens: Optional[int],
    ) -> None:
        """Raise an HTTPStatusError with TEI response details included."""
        if response.is_success:
            return

        message = (
            f"TEI embedding request failed with status={response.status_code} "
            f"batch_size={batch_size} batch_tokens={batch_tokens} "
            f"body={response.text!r}"
        )
        raise httpx.HTTPStatusError(
            message,
            request=response.request,
            response=response,
        )


def _iter_batches(
    inputs: Sequence[EmbeddingInput],
    *,
    max_batch_items: int,
    max_batch_tokens: int,
) -> Iterator[list[EmbeddingInput]]:
    """Yield batches capped by item count and estimated token count."""
    current_batch: list[EmbeddingInput] = []
    current_tokens = 0

    for item in inputs:
        token_count = item.token_count or 0
        if current_batch and (
            len(current_batch) >= max_batch_items
            or current_tokens + token_count > max_batch_tokens
        ):
            yield current_batch
            current_batch = []
            current_tokens = 0

        current_batch.append(item)
        current_tokens += token_count

    if current_batch:
        yield current_batch


def _sum_batch_tokens(batch: Sequence[EmbeddingInput]) -> Optional[int]:
    """Return the known token total for a batch when all counts are available."""
    total = 0
    for item in batch:
        if item.token_count is None:
            return None
        total += item.token_count
    return total


def _parse_single_embedding(payload: Any) -> EmbeddingVector:
    """Return one embedding vector from TEI single-input responses."""
    vectors = _parse_embedding_batch(payload)
    if len(vectors) != 1:
        raise ValueError(f"Expected one TEI embedding, received {len(vectors)}")
    return vectors[0]


def _parse_embedding_batch(payload: Any) -> list[EmbeddingVector]:
    """Parse and validate a TEI embedding batch response."""
    if not isinstance(payload, list):
        raise ValueError("TEI embedding response must be a list")
    return [_parse_embedding_vector(vector) for vector in payload]


def _parse_embedding_vector(value: Any) -> EmbeddingVector:
    """Parse and validate one TEI embedding vector."""
    if not isinstance(value, list):
        raise ValueError("TEI embedding vector must be a list")

    vector: EmbeddingVector = []
    for item in value:
        if not isinstance(item, (float, int)):
            raise ValueError("TEI embedding vector items must be numbers")
        vector.append(float(item))
    return vector
