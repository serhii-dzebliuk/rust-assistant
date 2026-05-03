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
    status = "completed"
    incomplete_details = None
    output = []


class FakeResponses:
    def __init__(self, response=None):
        self.response = response or FakeOpenAIResponse()
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAsyncOpenAI:
    def __init__(self, response=None):
        self.responses = FakeResponses(response=response)


class EmptyOpenAIResponse:
    output_text = ""
    model = "gpt-5-mini"
    usage = FakeUsage()
    status = "incomplete"
    incomplete_details = type("IncompleteDetails", (), {"reason": "max_output_tokens"})()
    output = []


@pytest.mark.asyncio
async def test_openai_llm_client_calls_responses_api_with_prepared_prompt():
    fake_client = FakeAsyncOpenAI()
    adapter = OpenAILLMClient(
        client=fake_client,
        model="gpt-5-mini",
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
            "model": "gpt-5-mini",
            "instructions": "system",
            "input": "user",
            "max_output_tokens": 500,
            "reasoning": {"effort": "minimal"},
            "store": False,
        }
    ]
    assert response.text == "Answer"
    assert response.model == "gpt-test"
    assert response.input_tokens == 12
    assert response.output_tokens == 8


@pytest.mark.asyncio
async def test_openai_llm_client_omits_reasoning_for_non_reasoning_models():
    fake_client = FakeAsyncOpenAI()
    adapter = OpenAILLMClient(
        client=fake_client,
        model="gpt-4o-mini",
        max_output_tokens=500,
    )

    await adapter.generate(
        LLMRequest(
            system_prompt="system",
            user_prompt="user",
            context=[],
        )
    )

    assert "reasoning" not in fake_client.responses.calls[0]


@pytest.mark.asyncio
async def test_openai_llm_client_rejects_empty_output_text():
    adapter = OpenAILLMClient(
        client=FakeAsyncOpenAI(response=EmptyOpenAIResponse()),
        model="gpt-5-mini",
        max_output_tokens=500,
    )

    with pytest.raises(ValueError, match="did not contain output text"):
        await adapter.generate(
            LLMRequest(
                system_prompt="system",
                user_prompt="user",
                context=[],
            )
        )
