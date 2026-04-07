import pytest

from rust_assistant.core.config import build_settings, get_settings

pytestmark = pytest.mark.unit


def test_build_settings_uses_defaults_for_optional_runtime_values():
    settings = build_settings({})

    assert settings.app.host == "0.0.0.0"
    assert settings.app.port == 8000
    assert settings.app.reload is False
    assert settings.app.api_mode == "stub"
    assert settings.dependencies.postgres == "not_configured"
    assert settings.dependencies.qdrant == "not_configured"


def test_build_settings_parses_explicit_values():
    settings = build_settings(
        {
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "RELOAD": "true",
            "API_MODE": "runtime",
            "POSTGRES_STATUS": "ready",
            "QDRANT_STATUS": "connected",
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "json",
            "POSTGRES_DB": "docs",
            "POSTGRES_USER": "app",
            "POSTGRES_PASSWORD": "secret",
            "DATABASE_URL": "postgres://app:secret@postgres:5432/docs",
            "QDRANT_URL": "http://qdrant:6333",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-5",
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_MODEL": "text-embedding-3-large",
            "PUBLIC_BASE_URL": "https://example.com",
        }
    )

    assert settings.app.host == "127.0.0.1"
    assert settings.app.port == 9000
    assert settings.app.reload is True
    assert settings.app.api_mode == "runtime"
    assert settings.dependencies.postgres == "ready"
    assert settings.dependencies.qdrant == "connected"
    assert settings.postgres.database == "docs"
    assert settings.postgres.user == "app"
    assert settings.postgres.password == "secret"
    assert settings.postgres.url == "postgres://app:secret@postgres:5432/docs"
    assert settings.qdrant.url == "http://qdrant:6333"
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-5"
    assert settings.llm.embedding_provider == "openai"
    assert settings.llm.embedding_model == "text-embedding-3-large"
    assert settings.logging.level == "DEBUG"
    assert settings.logging.format == "json"
    assert settings.proxy.public_base_url == "https://example.com"


def test_build_settings_rejects_invalid_boolean_values():
    try:
        build_settings({"RELOAD": "sometimes"})
    except ValueError as exc:
        assert "RELOAD" in str(exc)
    else:
        raise AssertionError("Expected build_settings to reject invalid boolean values")


def test_get_settings_uses_cache_and_can_be_cleared(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("PORT", "8123")
    settings = get_settings()

    assert settings.app.port == 8123

    get_settings.cache_clear()

