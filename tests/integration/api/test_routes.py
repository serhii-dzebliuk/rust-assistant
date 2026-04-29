from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from rust_assistant.bootstrap.api import create_app
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


@pytest.fixture
def search_use_case():
    return FakeSearchUseCase()


@pytest.fixture
def client(search_use_case: FakeSearchUseCase):
    container = SimpleNamespace(search_use_case=search_use_case)
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
            "k": 3,
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
    assert command.limit == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "async", "k": 0},
        {"query": "async", "k": 51},
    ],
)
def test_search_rejects_invalid_requests(client: TestClient, payload):
    response = client.post("/search", json=payload)

    assert response.status_code == 422


def test_system_endpoints_return_runtime_status(client: TestClient):
    health_response = client.get("/health")
    ready_response = client.get("/ready")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert ready_response.json() == {"status": "ready", "ready": True}


def test_unimplemented_chat_still_returns_not_implemented(client: TestClient):
    chat_response = client.post("/chat", json={"question": "What is async?"})

    assert chat_response.status_code == 501
    assert chat_response.json()["detail"] == "Not implemented"


def test_old_api_prefixed_routes_are_not_exposed(client: TestClient):
    assert client.get("/api/health").status_code == 404
    assert client.get("/api/ready").status_code == 404
    assert client.post("/api/search", json={"query": "async"}).status_code == 404
