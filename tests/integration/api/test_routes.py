import pytest
from fastapi.testclient import TestClient

from rust_assistant.bootstrap.api import create_app
from rust_assistant.bootstrap.settings import get_settings


pytestmark = pytest.mark.integration


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_MODE", "stub")
    monkeypatch.setenv("POSTGRES_STATUS", "ready")
    monkeypatch.setenv("QDRANT_STATUS", "ready")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


def test_declared_endpoints_return_not_implemented(client: TestClient):
    health_response = client.get("/health")
    ready_response = client.get("/ready")
    search_response = client.post("/search", json={"query": "async"})
    chat_response = client.post("/chat", json={"question": "What is async?"})

    assert health_response.status_code == 501
    assert ready_response.status_code == 501
    assert search_response.status_code == 501
    assert chat_response.status_code == 501
    assert health_response.json()["detail"] == "Not implemented"
    assert ready_response.json()["detail"] == "Not implemented"
    assert search_response.json()["detail"] == "Not implemented"
    assert chat_response.json()["detail"] == "Not implemented"


def test_old_api_prefixed_routes_are_not_exposed(client: TestClient):
    assert client.get("/api/health").status_code == 404
    assert client.get("/api/ready").status_code == 404
