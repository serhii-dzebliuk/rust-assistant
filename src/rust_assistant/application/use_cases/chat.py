"""Chat use-case orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rust_assistant.application.ports.llm_client import LLMClient, LLMRequest
from rust_assistant.application.ports.tokenizer import Tokenizer
from rust_assistant.application.services.prompt_builder import PromptBuilder
from rust_assistant.application.services.retrieval.models import (
    RetrievalRequest,
)
from rust_assistant.application.services.retrieval.pipeline import RetrievalPipeline

INSUFFICIENT_CONTEXT_ANSWER = (
    "I do not have enough relevant Rust documentation context to answer this question."
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ChatCommand:
    """Input command for the chat use case."""

    question: str


@dataclass(slots=True, frozen=True)
class ChatResult:
    """Chat use-case result."""

    answer: str


class ChatQuestionTooLargeError(ValueError):
    """Raised when a chat question exceeds the configured token budget."""


class ChatUseCase:
    """Generate a grounded answer for one independent chat question."""

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipeline,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        tokenizer: Tokenizer,
        retrieval_limit: int,
        reranking_limit: int,
        use_reranking: bool,
        max_query_tokens: int,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._prompt_builder = prompt_builder
        self._llm_client = llm_client
        self._tokenizer = tokenizer
        self._retrieval_limit = retrieval_limit
        self._reranking_limit = reranking_limit
        self._use_reranking = use_reranking
        self._max_query_tokens = max_query_tokens

    async def execute(self, command: ChatCommand) -> ChatResult:
        """Answer a chat question with retrieved Rust documentation context."""
        question = command.question.strip()
        query_tokens = self._tokenizer.count_tokens(question)
        logger.info(
            "Chat request accepted: question_chars=%s query_tokens=%s "
            "retrieval_limit=%s reranking_limit=%s use_reranking=%s",
            len(question),
            query_tokens,
            self._retrieval_limit,
            self._reranking_limit,
            self._use_reranking,
        )
        if query_tokens > self._max_query_tokens:
            logger.warning(
                "Chat question rejected: query_tokens=%s max_query_tokens=%s",
                query_tokens,
                self._max_query_tokens,
            )
            raise ChatQuestionTooLargeError(
                f"Question is too large; maximum is {self._max_query_tokens} tokens"
            )

        chunks = await self._retrieval_pipeline.retrieve(
            RetrievalRequest(
                query=question,
                retrieval_limit=self._retrieval_limit,
                reranking_limit=self._reranking_limit,
                use_reranking=self._use_reranking,
            )
        )
        logger.info("Chat retrieval completed: retrieved_chunks=%s", len(chunks))
        built_prompt = self._prompt_builder.build(question=question, chunks=chunks)
        logger.info(
            "Chat prompt built: selected_chunks=%s system_prompt_chars=%s "
            "user_prompt_chars=%s context_chunk_ids=%s context_titles=%s",
            len(built_prompt.context_chunks),
            len(built_prompt.system_prompt),
            len(built_prompt.user_prompt),
            [str(chunk.chunk_id) for chunk in built_prompt.context_chunks],
            [chunk.item_path or chunk.title for chunk in built_prompt.context_chunks],
        )
        if not built_prompt.context_chunks:
            logger.info("Chat returning insufficient-context fallback answer")
            return ChatResult(answer=INSUFFICIENT_CONTEXT_ANSWER)

        response = await self._llm_client.generate(
            LLMRequest(
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
                context=built_prompt.context_chunks,
            )
        )
        logger.info(
            "Chat answer generated: answer_chars=%s input_tokens=%s output_tokens=%s",
            len(response.text),
            response.input_tokens,
            response.output_tokens,
        )
        return ChatResult(answer=response.text)
