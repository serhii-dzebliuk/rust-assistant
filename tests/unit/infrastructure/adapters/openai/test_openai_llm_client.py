import pytest

from rust_assistant.application.ports.llm_client import LLMRequest
from rust_assistant.infrastructure.adapters.llm.openai.openai_llm_client import OpenAILLMClient

pytestmark = pytest.mark.unit


class FakeUsage:
    input_tokens = 12
    output_tokens = 8


class FakeOpenAIResponse:
    output_text = "Answer"
    model = "gpt-test"
    usage = FakeUsage()


class FakeResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOpenAIResponse()


class FakeAsyncOpenAI:
    def __init__(self):
        self.responses = FakeResponses()


@pytest.mark.asyncio
async def test_openai_llm_client_calls_responses_api_with_prepared_prompt():
    fake_client = FakeAsyncOpenAI()
    adapter = OpenAILLMClient(
        client=fake_client,
        model="gpt-test",
        max_output_tokens=500,
    )

    response = await adapter.generate(
        LLMRequest(
            system_prompt="system",
            user_prompt="user",
            context=[],
        )
    )

    assert fake_client.responses.calls == [
        {
            "model": "gpt-test",
            "instructions": "system",
            "input": "user",
            "max_output_tokens": 500,
            "store": False,
        }
    ]
    assert response.text == "Answer"
    assert response.model == "gpt-test"
    assert response.input_tokens == 12
    assert response.output_tokens == 8
