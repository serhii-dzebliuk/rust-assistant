from dataclasses import dataclass
from typing import Optional, Protocol

from rust_assistant.application.services.retrieval.models import RetrievedChunk


@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str
    context: list[RetrievedChunk]


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class LLMClient(Protocol):
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse: ...
