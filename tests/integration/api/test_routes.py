import pytest
from fastapi.testclient import TestClient

from rust_assistant.core.config import get_settings
from rust_assistant.main import create_app


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


def test_health_and_ready_are_served_without_api_prefix(client: TestClient):
    health_response = client.get("/health")
    ready_response = client.get("/ready")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert ready_response.json()["status"] == "ready"


def test_old_api_prefixed_routes_are_not_exposed(client: TestClient):
    assert client.get("/api/health").status_code == 404
    assert client.get("/api/ready").status_code == 404
