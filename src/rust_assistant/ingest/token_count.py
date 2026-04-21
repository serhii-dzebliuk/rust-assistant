"""Token counting helpers for ingest chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rust_assistant.ingest.entities import Chunk


class Tokenizer(Protocol):
    """Minimal tokenizer protocol used by the ingest pipeline."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token ids."""


@dataclass(slots=True, frozen=True)
class ChunkTokenCounter:
    """Count chunk tokens with the tokenizer used by the embedding model."""

    model_name: str
    tokenizer: Tokenizer

    @classmethod
    def from_model_name(cls, model_name: str) -> "ChunkTokenCounter":
        """Load an AutoTokenizer for the configured embedding model."""
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("EMBEDDING_MODEL must be configured before counting chunk tokens")

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers must be installed to compute chunk token counts"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(normalized_model_name)
        return cls(model_name=normalized_model_name, tokenizer=tokenizer)

    def count_text_tokens(self, text: str) -> int:
        """Return the number of model tokens in chunk text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def with_token_counts(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks copied with tokenizer-derived token counts."""
        return [
            chunk.model_copy(update={"token_count": self.count_text_tokens(chunk.text)})
            for chunk in chunks
        ]


__all__ = ["ChunkTokenCounter"]
