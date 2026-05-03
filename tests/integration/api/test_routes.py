from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from rust_assistant.bootstrap.api import create_app
from rust_assistant.application.use_cases.chat import ChatQuestionTooLargeError, ChatResult
from rust_assistant.application.use_cases.search import SearchResult, SearchResultHit

pytestmark = pytest.mark.integration


class FakeSearchUseCase:
    def __init__(self):
        self.commands = []

    async def execute(self, command):
        self.commands.append(command)
        return SearchResult(
            query=command.query,
            hits=[
                SearchResultHit(
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
                    score=0.91,
                    text="Async returns a Future.",
                ),
            ],
        )


class FakeChatUseCase:
    def __init__(self):
        self.commands = []
        self.error = None

    async def execute(self, command):
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return ChatResult(answer=f"Answer for {command.question}")


@pytest.fixture
def search_use_case():
    return FakeSearchUseCase()


@pytest.fixture
def chat_use_case():
    return FakeChatUseCase()


@pytest.fixture
def client(search_use_case: FakeSearchUseCase, chat_use_case: FakeChatUseCase):
    container = SimpleNamespace(search_use_case=search_use_case, chat_use_case=chat_use_case)
    with TestClient(create_app(container=container)) as test_client:
        yield test_client


def test_search_maps_request_to_use_case_and_returns_enriched_hits(
    client: TestClient,
    search_use_case: FakeSearchUseCase,
):
    response = client.post(
        "/search",
        json={
            "query": " async ",
            "retrieval_limit": 30,
            "reranking_limit": 3,
            "use_reranking": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "async",
        "total_results": 1,
        "results": [
            {
                "chunk_id": "11111111-1111-4111-8111-111111111111",
                "document_id": "22222222-2222-4222-8222-222222222222",
                "title": "std::keyword::async",
                "source_path": "std/keyword.async.html",
                "url": "https://doc.rust-lang.org/std/keyword.async.html",
                "section": "Keyword async",
                "item_path": "std::keyword::async",
                "crate": "std",
                "item_type": "keyword",
                "rust_version": "1.91.1",
                "score": 0.91,
                "text": "Async returns a Future.",
            }
        ],
    }
    command = search_use_case.commands[0]
    assert command.query == "async"
    assert command.retrieval_limit == 30
    assert command.reranking_limit == 3
    assert command.use_reranking is False


def test_search_uses_default_retrieval_and_reranking_limits(
    client: TestClient,
    search_use_case: FakeSearchUseCase,
):
    response = client.post("/search", json={"query": "async"})

    assert response.status_code == 200
    command = search_use_case.commands[0]
    assert command.retrieval_limit == 20
    assert command.reranking_limit == 10
    assert command.use_reranking is True


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "async", "retrieval_limit": 0},
        {"query": "async", "retrieval_limit": 101},
        {"query": "async", "reranking_limit": 0},
        {"query": "async", "reranking_limit": 101},
        {"query": "async", "retrieval_limit": 5, "reranking_limit": 6},
        {"query": "async", "k": 5},
    ],
)
def test_search_rejects_invalid_requests(client: TestClient, payload):
    response = client.post("/search", json=payload)

    assert response.status_code == 422


def test_health_endpoint_returns_runtime_status(client: TestClient):
    health_response = client.get("/health")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert client.get("/ready").status_code == 404


def test_chat_maps_request_to_use_case_and_returns_answer_only(
    client: TestClient,
    chat_use_case: FakeChatUseCase,
):
    chat_response = client.post("/chat", json={"question": " What is async? "})

    assert chat_response.status_code == 200
    assert chat_response.json() == {"answer": "Answer for What is async?"}
    assert chat_use_case.commands[0].question == "What is async?"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "async", "k": 5},
        {"question": "async", "retrieval_limit": 20},
        {"question": "async", "use_reranking": False},
    ],
)
def test_chat_rejects_invalid_requests(client: TestClient, payload):
    response = client.post("/chat", json=payload)

    assert response.status_code == 422


def test_chat_returns_422_for_too_large_question(
    client: TestClient,
    chat_use_case: FakeChatUseCase,
):
    chat_use_case.error = ChatQuestionTooLargeError("Question is too large; maximum is 1000 tokens")

    response = client.post("/chat", json={"question": "async"})

    assert response.status_code == 422
    assert response.json()["detail"] == "Question is too large; maximum is 1000 tokens"


def test_chat_returns_503_when_not_configured(search_use_case: FakeSearchUseCase):
    container = SimpleNamespace(search_use_case=search_use_case, chat_use_case=None)
    with TestClient(create_app(container=container)) as test_client:
        response = test_client.post("/chat", json={"question": "async"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Chat is not configured"


def test_old_api_prefixed_routes_are_not_exposed(client: TestClient):
    assert client.get("/api/health").status_code == 404
    assert client.get("/api/ready").status_code == 404
    assert client.post("/api/search", json={"query": "async"}).status_code == 404
