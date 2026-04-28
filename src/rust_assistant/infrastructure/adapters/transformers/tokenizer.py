"""Transformers-backed tokenizer adapter."""

from __future__ import annotations


class TransformersTokenizer:
    """Count tokens with the tokenizer used by the configured embedding model."""

    def __init__(self, model_name: str):
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("EMBEDDING_MODEL must be configured before counting chunk tokens")

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers must be installed to compute chunk token counts"
            ) from exc

        self._model_name = normalized_model_name
        self._tokenizer = AutoTokenizer.from_pretrained(normalized_model_name)

    @property
    def model_name(self) -> str:
        """Return the tokenizer model name."""
        return self._model_name

    def count_tokens(self, text: str) -> int:
        """Return the number of model tokens in text."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))


__all__ = ["TransformersTokenizer"]
