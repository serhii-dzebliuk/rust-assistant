"""Prompt building helpers for the serving runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .retriever import RetrievedChunk


@dataclass(slots=True, frozen=True)
class PromptPayload:
    """Structured prompt assembled for the language model."""

    system_prompt: str
    user_prompt: str
    context_blocks: list[str] = field(default_factory=list)
    rendered_prompt: str = ""


class PromptBuilder:
    """Build prompts from a user question and retrieved Rust docs context."""

    SYSTEM_PROMPT = (
        "You are a Rust documentation assistant. Answer using the retrieved "
        "context when it is available, and be honest when the real runtime "
        "is still operating in stub mode."
    )

    def build_chat_prompt(
        self,
        *,
        question: str,
        context_chunks: Sequence[RetrievedChunk],
    ) -> PromptPayload:
        """Render a simple prompt from the question and retrieved sources."""
        context_blocks = [
            self._format_context_block(index + 1, chunk)
            for index, chunk in enumerate(context_chunks)
        ]
        context_text = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."
        user_prompt = f"Question: {question}\n\nContext:\n{context_text}"
        rendered_prompt = f"System:\n{self.SYSTEM_PROMPT}\n\nUser:\n{user_prompt}"
        return PromptPayload(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context_blocks=context_blocks,
            rendered_prompt=rendered_prompt,
        )

    @staticmethod
    def _format_context_block(rank: int, chunk: RetrievedChunk) -> str:
        """Render one retrieved chunk for prompt inclusion."""
        label = chunk.title or chunk.item_path or chunk.source_path
        snippet = chunk.snippet or "(empty snippet)"
        return f"[{rank}] {label}\nSource: {chunk.source_path}\nContent: {snippet}"
