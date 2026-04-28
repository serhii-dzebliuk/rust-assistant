"""Port for model-compatible token counting."""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    """Count model tokens for plain text."""

    def count_tokens(self, text: str) -> int:
        """Return the number of model tokens in text."""
        ...
