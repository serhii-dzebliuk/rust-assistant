import httpx
from typing import Sequence

from rust_assistant.application.ports.embedding_client import EmbeddingInput, EmbeddingVector


class TeiEmbeddingClient():
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
        response = await self._client.post(
                f"{self._base_url}/embed",
                json={
                    "inputs": text,
                    "normalize": self._normalize,
                },
            )
        response.raise_for_status()
        return response.json()
    

    async def embed_texts(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingVector]:
        vectors: list[EmbeddingVector] = []

        for batch in self._build_batches(inputs):
            batch_vectors = await self._embed_batch(batch)
            vectors.extend(batch_vectors)

        return vectors
    

    async def _embed_batch(self, batch: Sequence[str]) -> list[EmbeddingVector]:
        response = await self._client.post(
            f"{self._base_url}/embed",
            json={
                "inputs": batch,
                "normalize": self._normalize,
            },
        )
        response.raise_for_status()
        return response.json()
    

    def _build_batches(self, inputs: Sequence[EmbeddingInput]) -> list[list[str]]:
        batches: list[list[str]] = []

        current_batch: list[str] = []
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

            current_batch.append(item.text)
            current_tokens += token_count

        if current_batch:
            batches.append(current_batch)

        return batches