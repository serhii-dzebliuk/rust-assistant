from uuid import UUID

import pytest

from rust_assistant.application.services.retrieval.models import RetrievedChunk
from rust_assistant.application.use_cases.chat import ChatCommand, ChatUseCase

pytestmark = pytest.mark.unit


class FakeRetrievalPipeline:
    def __init__(self, *, chunks=None):
        self.chunks = chunks or []
        self.requests = []

    async def retrieve(self, request):
        self.requests.append(request)
        return self.chunks


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID("11111111-1111-4111-8111-111111111111"),
        document_id=UUID("22222222-2222-4222-8222-222222222222"),
        title="std::keyword::async",
        source_path="std/keyword.async.html",
        url="https://doc.rust-lang.org/std/keyword.async.html",
        section="Keyword async",
        item_path="std::keyword::async",
        crate="std",
        item_type="keyword",
        rust_version="1.91.1",
        score=0.98,
        text="Async returns a Future.",
    )


@pytest.mark.asyncio
async def test_chat_maps_command_to_retrieval_request():
    retrieval_pipeline = FakeRetrievalPipeline()

    await ChatUseCase(retrieval_pipeline=retrieval_pipeline).execute(
        ChatCommand(
            question=" What is async? ",
            retrieval_limit=30,
            reranking_limit=3,
            use_reranking=False,
        )
    )

    request = retrieval_pipeline.requests[0]
    assert request.query == "What is async?"
    assert request.retrieval_limit == 30
    assert request.reranking_limit == 3
    assert request.use_reranking is False


@pytest.mark.asyncio
async def test_chat_returns_retrieved_sources_without_llm_answer():
    result = await ChatUseCase(
        retrieval_pipeline=FakeRetrievalPipeline(chunks=[_chunk()])
    ).execute(ChatCommand(question="What is async?"))

    assert result.answer == ""
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == UUID("11111111-1111-4111-8111-111111111111")
    assert result.sources[0].document_id == UUID("22222222-2222-4222-8222-222222222222")
    assert result.sources[0].title == "std::keyword::async"
    assert result.sources[0].source_path == "std/keyword.async.html"
    assert result.sources[0].url == "https://doc.rust-lang.org/std/keyword.async.html"
    assert result.sources[0].section == "Keyword async"
    assert result.sources[0].item_path == "std::keyword::async"
    assert result.sources[0].crate == "std"
    assert result.sources[0].item_type == "keyword"
    assert result.sources[0].rust_version == "1.91.1"
    assert result.sources[0].score == 0.98
    assert result.sources[0].text == "Async returns a Future."
