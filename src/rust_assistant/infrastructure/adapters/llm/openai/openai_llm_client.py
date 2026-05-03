"""OpenAI SDK-backed LLM adapter."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from rust_assistant.application.ports.llm_client import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

REASONING_MODEL_PREFIXES = ("gpt-5", "o")


class OpenAILLMClient:
    """Generate answers through the OpenAI Responses API."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        max_output_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one answer from a prepared prompt."""
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "max_output_tokens": self._max_output_tokens,
            "store": False,
        }
        if _uses_reasoning_config(self._model):
            create_kwargs["reasoning"] = {"effort": "minimal"}

        logger.info(
            "OpenAI request prepared: model=%s max_output_tokens=%s "
            "reasoning_effort=%s system_prompt_chars=%s user_prompt_chars=%s "
            "context_chunks=%s",
            self._model,
            self._max_output_tokens,
            _reasoning_effort(create_kwargs),
            len(request.system_prompt),
            len(request.user_prompt),
            len(request.context),
        )
        response = await self._client.responses.create(**create_kwargs)
        usage = getattr(response, "usage", None)
        text = getattr(response, "output_text", "").strip()
        _log_response_summary(response, text)
        if not text:
            _log_empty_response(response)
            raise ValueError("OpenAI response did not contain output text")

        return LLMResponse(
            text=text,
            model=getattr(response, "model", self._model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )


def _uses_reasoning_config(model: str) -> bool:
    normalized_model = model.strip().lower()
    return normalized_model.startswith(REASONING_MODEL_PREFIXES)


def _log_empty_response(response: object) -> None:
    usage = getattr(response, "usage", None)
    output_tokens_details = getattr(usage, "output_tokens_details", None)
    incomplete_details = getattr(response, "incomplete_details", None)
    logger.warning(
        "OpenAI response contained no output text: "
        "status=%s incomplete_reason=%s output_types=%s input_tokens=%s "
        "output_tokens=%s reasoning_tokens=%s",
        getattr(response, "status", None),
        getattr(incomplete_details, "reason", None),
        _output_types(response),
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
        getattr(output_tokens_details, "reasoning_tokens", None),
    )


def _log_response_summary(response: object, text: str) -> None:
    usage = getattr(response, "usage", None)
    output_tokens_details = getattr(usage, "output_tokens_details", None)
    incomplete_details = getattr(response, "incomplete_details", None)
    logger.info(
        "OpenAI response received: model=%s status=%s incomplete_reason=%s "
        "output_types=%s output_text_chars=%s input_tokens=%s output_tokens=%s "
        "reasoning_tokens=%s",
        getattr(response, "model", None),
        getattr(response, "status", None),
        getattr(incomplete_details, "reason", None),
        _output_types(response),
        len(text),
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
        getattr(output_tokens_details, "reasoning_tokens", None),
    )


def _output_types(response: object) -> list[str]:
    output_items = getattr(response, "output", None) or []
    return [str(getattr(item, "type", "unknown")) for item in output_items]


def _reasoning_effort(create_kwargs: dict[str, object]) -> object:
    reasoning = create_kwargs.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    return reasoning.get("effort")
