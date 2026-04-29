"""TEI embedding adapter implementation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from rust_assistant.application.ports.embedding_client import EmbeddingInput, EmbeddingVector

logger = logging.getLogger(__name__)


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
        return response.json()

    async def embed_texts(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingVector]:
        """Embed text inputs in token-aware batches."""
        total_inputs = len(inputs)
        vectors: list[EmbeddingVector] = []
        if not inputs:
            return vectors

        logger.info("Embedding chunks: 0/%s (0.0%%)", total_inputs)

        for batch in self._build_batches(inputs):
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
        return response.json()

    async def _post_embed(self, inputs: Any) -> httpx.Response:
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
        batch_tokens: int | None,
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

    def _build_batches(self, inputs: Sequence[EmbeddingInput]) -> list[list[EmbeddingInput]]:
        """Build batches capped by item count and estimated token count."""
        batches: list[list[EmbeddingInput]] = []

        current_batch: list[EmbeddingInput] = []
        current_tokens = 0

        for item in inputs:
            token_count = item.token_count

            if token_count is None:
                # fallback: batch only by item count
                token_count = 0

            if (
                current_batch
                and (
                    len(current_batch) >= self._max_batch_items
                    or current_tokens + token_count > self._max_batch_tokens
                )
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(item)
            current_tokens += token_count

        if current_batch:
            batches.append(current_batch)

        return batches


def _sum_batch_tokens(batch: Sequence[EmbeddingInput]) -> int | None:
    """Return the known token total for a batch when all counts are available."""
    total = 0
    for item in batch:
        if item.token_count is None:
            return None
        total += item.token_count
    return total
