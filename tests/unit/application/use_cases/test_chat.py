from uuid import UUID

import pytest

from rust_assistant.application.ports.llm_client import LLMResponse
from rust_assistant.application.services.prompt_builder import PromptBuilder
from rust_assistant.application.services.retrieval.models import RetrievedChunk
from rust_assistant.application.use_cases.chat import (
    ChatCommand,
    ChatQuestionTooLargeError,
    ChatUseCase,
    INSUFFICIENT_CONTEXT_ANSWER,
)

pytestmark = pytest.mark.unit


class FakeRetrievalPipeline:
    def __init__(self, *, chunks=None):
        self.chunks = chunks or []
        self.requests = []

    async def retrieve(self, request):
        self.requests.append(request)
        return self.chunks


class FakeLLMClient:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return LLMResponse(text="Use async to create a Future.", model="fake-model")


class FakeTokenizer:
    def __init__(self, *, counts=None, default_count=1):
        self.counts = counts or {}
        self.default_count = default_count
        self.texts = []

    def count_tokens(self, text):
        self.texts.append(text)
        return self.counts.get(text, self.default_count)


def _chunk(
    *,
    chunk_id="11111111-1111-4111-8111-111111111111",
    title="std::keyword::async",
    item_path="std::keyword::async",
    crate="std",
    text="Async returns a Future.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(chunk_id),
        document_id=UUID("22222222-2222-4222-8222-222222222222"),
        title=title,
        source_path="std/keyword.async.html",
        url="https://doc.rust-lang.org/std/keyword.async.html",
        section="Keyword async",
        item_path=item_path,
        crate=crate,
        item_type="keyword",
        rust_version="1.91.1",
        score=0.98,
        text=text,
    )


def _use_case(
    *,
    retrieval_pipeline=None,
    tokenizer=None,
    llm_client=None,
    max_query_tokens=1000,
) -> tuple[ChatUseCase, FakeRetrievalPipeline, FakeLLMClient]:
    fake_retrieval_pipeline = retrieval_pipeline or FakeRetrievalPipeline()
    fake_tokenizer = tokenizer or FakeTokenizer()
    fake_llm_client = llm_client or FakeLLMClient()
    return (
        ChatUseCase(
            retrieval_pipeline=fake_retrieval_pipeline,
            prompt_builder=PromptBuilder(
                tokenizer=fake_tokenizer,
                max_context_tokens=2500,
            ),
            llm_client=fake_llm_client,
            tokenizer=fake_tokenizer,
            retrieval_limit=30,
            reranking_limit=3,
            use_reranking=False,
            max_query_tokens=max_query_tokens,
        ),
        fake_retrieval_pipeline,
        fake_llm_client,
    )


@pytest.mark.asyncio
async def test_chat_uses_configured_retrieval_settings_and_generates_answer():
    use_case, retrieval_pipeline, llm_client = _use_case(
        retrieval_pipeline=FakeRetrievalPipeline(chunks=[_chunk()])
    )

    result = await use_case.execute(ChatCommand(question=" What is async? "))

    request = retrieval_pipeline.requests[0]
    assert request.query == "What is async?"
    assert request.retrieval_limit == 30
    assert request.reranking_limit == 3
    assert request.use_reranking is False
    assert result.answer == "Use async to create a Future."
    assert "Question:\nWhat is async?" in llm_client.requests[0].user_prompt


@pytest.mark.asyncio
async def test_chat_rejects_large_question_before_retrieval_and_llm():
    use_case, retrieval_pipeline, llm_client = _use_case(
        tokenizer=FakeTokenizer(counts={"too large": 2}),
        max_query_tokens=1,
    )

    with pytest.raises(ChatQuestionTooLargeError, match="maximum is 1 tokens"):
        await use_case.execute(ChatCommand(question="too large"))

    assert retrieval_pipeline.requests == []
    assert llm_client.requests == []


@pytest.mark.asyncio
async def test_chat_skips_llm_when_retrieval_returns_no_context():
    use_case, _, llm_client = _use_case()

    result = await use_case.execute(ChatCommand(question="What is async?"))

    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert llm_client.requests == []
