"""OpenAI SDK-backed LLM adapter."""

from __future__ import annotations

from openai import AsyncOpenAI

from rust_assistant.application.ports.llm_client import LLMRequest, LLMResponse


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
        response = await self._client.responses.create(
            model=self._model,
            instructions=request.system_prompt,
            input=request.user_prompt,
            max_output_tokens=self._max_output_tokens,
            store=False,
        )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=getattr(response, "output_text", ""),
            model=getattr(response, "model", self._model),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
