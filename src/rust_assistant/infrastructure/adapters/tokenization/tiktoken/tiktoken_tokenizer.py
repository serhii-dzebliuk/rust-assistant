"""tiktoken-backed tokenizer adapter."""

from __future__ import annotations


class TiktokenTokenizer:
    """Count tokens with an OpenAI-compatible tokenizer."""

    def __init__(self, model_name: str) -> None:
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("OPENAI_MODEL must be configured before counting chat tokens")

        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("tiktoken must be installed to count chat tokens") from exc

        try:
            self._encoding = tiktoken.encoding_for_model(normalized_model_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("o200k_base")

    def count_tokens(self, text: str) -> int:
        """Return the number of model tokens in text."""
        return len(self._encoding.encode(text))


__all__ = ["TiktokenTokenizer"]
