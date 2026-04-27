"""Port for enriching chunks with tokenizer-derived token counts."""

from __future__ import annotations

from typing import Protocol

from rust_assistant.domain.entities.chunks import Chunk


class ChunkTokenCounterPort(Protocol):
    """Compute token counts for chunks before persistence."""

    def with_token_counts(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks copied with computed token counts."""
        ...
