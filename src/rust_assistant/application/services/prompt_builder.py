"""Prompt construction for grounded chat answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rust_assistant.application.ports.tokenizer import Tokenizer
from rust_assistant.application.services.retrieval.models import RetrievedChunk

SYSTEM_PROMPT = """You are a technical assistant for the Rust programming language.

Your job is to answer the user's question using the provided Rust documentation excerpts as the primary source of truth.

Rules:
- If the answer is supported by the provided material, answer directly.
- If the provided material is insufficient, say that you do not have enough information to answer reliably.
- Do not invent APIs, behavior, guarantees, examples, or version-specific details.
- Use general Rust knowledge only to clarify wording or explain basic concepts; do not use it to override or extend the provided material.

Behavior:
- Prefer precise, practical, and structured answers.
- Use bullet points or steps when they make the answer clearer.
- Include minimal code examples only when they are directly supported by the provided material.
- For troubleshooting questions, suggest likely causes and fixes only when they are supported by the provided material.

Source handling:
- If provided sources conflict, mention that the available material appears inconsistent and prefer the most specific source.
- Prefer item-specific documentation over broad overview text when both are available.

Restrictions:
- Do not mention embeddings, retrieval, reranking, chunks, prompts, or system internals.
- Do not explicitly refer to "the context" in the final answer.

Language:
- Answer in the same language as the user."""

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class BuiltPrompt:
    """A prompt split into model instructions and user-visible task input."""

    system_prompt: str
    user_prompt: str
    context_chunks: list[RetrievedChunk]


class PromptBuilder:
    """Build LLM prompts from a question and token-limited retrieval context."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        max_context_tokens: int,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_context_tokens = max_context_tokens
        self._system_prompt = system_prompt

    def build(self, *, question: str, chunks: list[RetrievedChunk]) -> BuiltPrompt:
        """Return model instructions and a user prompt with token-limited context."""
        selected_chunks, context_tokens = self._select_context_chunks(chunks)
        logger.info(
            "Prompt context selected: candidate_chunks=%s selected_chunks=%s "
            "context_tokens=%s max_context_tokens=%s",
            len(chunks),
            len(selected_chunks),
            context_tokens,
            self._max_context_tokens,
        )
        return BuiltPrompt(
            system_prompt=self._system_prompt,
            user_prompt=_format_user_prompt(
                question=question,
                chunks=selected_chunks,
                tokenizer=self._tokenizer,
            ),
            context_chunks=selected_chunks,
        )

    def _select_context_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], int]:
        selected_chunks: list[RetrievedChunk] = []
        total_tokens = 0
        for chunk in chunks:
            token_count = self._tokenizer.count_tokens(chunk.text)
            if total_tokens + token_count > self._max_context_tokens:
                logger.info(
                    "Prompt context budget reached: next_chunk_id=%s "
                    "next_chunk_tokens=%s current_context_tokens=%s max_context_tokens=%s",
                    chunk.chunk_id,
                    token_count,
                    total_tokens,
                    self._max_context_tokens,
                )
                break
            selected_chunks.append(chunk)
            total_tokens += token_count
        return selected_chunks, total_tokens


def _format_user_prompt(
    *,
    question: str,
    chunks: list[RetrievedChunk],
    tokenizer: Tokenizer,
) -> str:
    context_blocks = [
        _format_context_block(index=index, chunk=chunk, tokenizer=tokenizer)
        for index, chunk in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)
    return f"Question:\n{question}\n\nContext:\n{context}"


def _format_context_block(
    *,
    index: int,
    chunk: RetrievedChunk,
    tokenizer: Tokenizer,
) -> str:
    title = chunk.item_path or chunk.title
    crate = chunk.crate or "unknown"
    token_count = tokenizer.count_tokens(chunk.text)
    return f"[{index}] {title} | {crate} | token_count={token_count}\n{chunk.text}"
