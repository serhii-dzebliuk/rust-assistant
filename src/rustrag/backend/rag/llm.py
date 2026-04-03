"""Language model abstraction for the serving runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .prompt import PromptPayload
from .retriever import RetrievedChunk


@dataclass(slots=True, frozen=True)
class LLMResult:
    """Normalized response returned by the runtime LLM client."""

    answer: str
    confidence: str = "unknown"
    model_name: str = "stub-llm"


class LLMClient(Protocol):
    """Interface for language model providers used by chat runtime."""

    def answer(
        self,
        *,
        prompt: PromptPayload,
        question: str,
        context_chunks: Sequence[RetrievedChunk],
    ) -> LLMResult:
        """Generate an answer for the given prompt and retrieved context."""


class StubLLMClient:
    """Deterministic placeholder model used until a real LLM is configured."""

    def __init__(self, *, model_name: str = "stub-llm") -> None:
        self._model_name = model_name

    def answer(
        self,
        *,
        prompt: PromptPayload,
        question: str,
        context_chunks: Sequence[RetrievedChunk],
    ) -> LLMResult:
        """Return a stable answer that keeps the API usable during scaffolding."""
        _ = prompt, question
        if context_chunks:
            titles = ", ".join(chunk.title for chunk in context_chunks[:2] if chunk.title)
            answer = (
                "Stub LLM response is active. Real Postgres + Qdrant-backed "
                "logic is not connected yet. "
                f"Retrieved context is available for: {titles or 'the current query'}."
            )
        else:
            answer = (
                "Chat API stub is up. Real Postgres + Qdrant-backed logic "
                "is not connected yet."
            )

        return LLMResult(
            answer=answer,
            confidence="unknown",
            model_name=self._model_name,
        )
