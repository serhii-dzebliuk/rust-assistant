import pytest

from rust_assistant.bootstrap.container import RuntimeConfigurationError, build_container
from rust_assistant.bootstrap.settings import build_settings


pytestmark = pytest.mark.unit


def test_build_container_is_lightweight_by_default():
    container = build_container(settings=build_settings({}))

    assert container.search_use_case is None
    assert container.db_engine is None
    assert container.http_client is None
    assert container.qdrant_client is None


def test_build_container_requires_search_runtime_settings_when_enabled():
    with pytest.raises(RuntimeConfigurationError, match="DATABASE_URL"):
        build_container(settings=build_settings({}), include_search=True)


def test_build_container_requires_reranker_base_url_when_search_enabled():
    settings = build_settings(
        {
            "DATABASE_URL": "postgresql+asyncpg://app:secret@postgres:5432/docs",
            "QDRANT_URL": "http://qdrant:6333",
            "QDRANT_VECTOR_SIZE": "768",
            "EMBEDDING_BASE_URL": "http://tei-embedder:80",
        }
    )

    with pytest.raises(RuntimeConfigurationError, match="RERANKER_BASE_URL"):
        build_container(settings=settings, include_search=True)
