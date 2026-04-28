from dataclasses import dataclass
from typing import Protocol, Sequence


EmbeddingVector = list[float]


@dataclass(frozen=True)
class EmbeddingInput:
    text: str
    token_count: int | None = None


class EmbeddingClient(Protocol):
    async def embed_text(self, text: str) -> EmbeddingVector:
        ...

    async def embed_texts(self, texts: Sequence[EmbeddingInput]) -> list[EmbeddingVector]:
        ...