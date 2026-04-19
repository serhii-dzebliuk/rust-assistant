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
    assert settings.postgres.echo is False
    assert settings.postgres.pool_size == 10
    assert settings.postgres.max_overflow == 10
    assert settings.ingest.raw_docs_dir is None
    assert settings.ingest.max_chunk_chars == 1400
    assert settings.ingest.min_chunk_chars == 180


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
            "DATABASE_URL": "postgresql+asyncpg://app:secret@postgres:5432/docs",
            "POSTGRES_ECHO": "true",
            "POSTGRES_POOL_SIZE": "20",
            "POSTGRES_MAX_OVERFLOW": "5",
            "QDRANT_URL": "http://qdrant:6333",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-5",
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_MODEL": "text-embedding-3-large",
            "RUST_DOCS_RAW_DIR": "D:\\rust-docs",
            "INGEST_MAX_CHUNK_CHARS": "1200",
            "INGEST_MIN_CHUNK_CHARS": "120",
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
    assert settings.postgres.url == "postgresql+asyncpg://app:secret@postgres:5432/docs"
    assert settings.postgres.echo is True
    assert settings.postgres.pool_size == 20
    assert settings.postgres.max_overflow == 5
    assert settings.qdrant.url == "http://qdrant:6333"
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-5"
    assert settings.llm.embedding_provider == "openai"
    assert settings.llm.embedding_model == "text-embedding-3-large"
    assert str(settings.ingest.raw_docs_dir) == "D:\\rust-docs"
    assert settings.ingest.max_chunk_chars == 1200
    assert settings.ingest.min_chunk_chars == 120
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


def test_build_settings_rejects_min_chunk_chars_above_max():
    with pytest.raises(ValueError, match="INGEST_MIN_CHUNK_CHARS"):
        build_settings(
            {
                "INGEST_MAX_CHUNK_CHARS": "100",
                "INGEST_MIN_CHUNK_CHARS": "101",
            }
        )


def test_get_settings_uses_cache_and_can_be_cleared(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("PORT", "8123")
    settings = get_settings()

    assert settings.app.port == 8123

    get_settings.cache_clear()
