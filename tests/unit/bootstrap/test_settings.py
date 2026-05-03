import pytest

from rust_assistant.bootstrap.settings import build_settings, get_settings

pytestmark = pytest.mark.unit


def test_build_settings_uses_defaults_for_optional_runtime_values():
    settings = build_settings({})

    assert settings.app.host == "0.0.0.0"
    assert settings.app.port == 8000
    assert settings.postgres.echo is False
    assert settings.postgres.pool_size == 10
    assert settings.postgres.max_overflow == 10
    assert settings.qdrant.collection_name == "rust-docs"
    assert settings.qdrant.vector_size is None
    assert settings.qdrant.distance == "cosine"
    assert settings.qdrant.upsert_batch_size == 256
    assert settings.embedding.model is None
    assert settings.embedding.base_url is None
    assert settings.embedding.normalize is True
    assert settings.embedding.pooling == "mean"
    assert settings.embedding.max_batch_tokens == 4096
    assert settings.embedding.max_batch_items == 64
    assert settings.embedding.max_concurrent_requests == 8
    assert settings.embedding.request_timeout_seconds == 120.0
    assert settings.reranker.model is None
    assert settings.reranker.base_url is None
    assert settings.reranker.max_batch_items == 32
    assert settings.openai.model is None
    assert settings.openai.api_key is None
    assert settings.openai.max_output_tokens == 500
    assert settings.openai.temperature == 0.2
    assert settings.openai.request_timeout_seconds == 60.0
    assert settings.chat.retrieval_limit == 20
    assert settings.chat.reranking_limit == 5
    assert settings.chat.use_reranking is True
    assert settings.chat.max_query_tokens == 1000
    assert settings.chat.max_context_tokens == 2500
    assert settings.ingest.raw_docs_dir is None
    assert settings.ingest.max_chunk_chars == 1400
    assert settings.ingest.min_chunk_chars == 180


def test_build_settings_parses_explicit_values():
    settings = build_settings(
        {
            "HOST": "127.0.0.1",
            "PORT": "9000",
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
            "QDRANT_COLLECTION_NAME": "rust-docs-v2",
            "QDRANT_VECTOR_SIZE": "768",
            "QDRANT_DISTANCE": "dot",
            "QDRANT_UPSERT_BATCH_SIZE": "128",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-5",
            "OPENAI_MAX_OUTPUT_TOKENS": "300",
            "OPENAI_TEMPERATURE": "0.1",
            "OPENAI_REQUEST_TIMEOUT_SECONDS": "45.5",
            "CHAT_RETRIEVAL_LIMIT": "30",
            "CHAT_RERANKING_LIMIT": "4",
            "CHAT_USE_RERANKING": "false",
            "CHAT_MAX_QUERY_TOKENS": "900",
            "CHAT_MAX_CONTEXT_TOKENS": "2400",
            "EMBEDDING_MODEL": "microsoft/harrier-oss-v1-270m",
            "EMBEDDING_BASE_URL": "http://tei:80",
            "EMBEDDING_NORMALIZE": "false",
            "EMBEDDING_MAX_BATCH_ITEMS": "32",
            "EMBEDDING_REQUEST_TIMEOUT_SECONDS": "180.5",
            "EMBEDDING_POOLING": "mean",
            "EMBEDDING_MAX_BATCH_TOKENS": "8192",
            "EMBEDDING_MAX_CONCURRENT_REQUESTS": "12",
            "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "RERANKER_BASE_URL": "http://tei-reranker:80",
            "RERANKER_MAX_BATCH_ITEMS": "16",
            "RUST_DOCS_RAW_DIR": "D:\\rust-docs",
            "INGEST_MAX_CHUNK_CHARS": "1200",
            "INGEST_MIN_CHUNK_CHARS": "120",
            "PUBLIC_BASE_URL": "https://example.com",
        }
    )

    assert settings.app.host == "127.0.0.1"
    assert settings.app.port == 9000
    assert settings.postgres.database == "docs"
    assert settings.postgres.user == "app"
    assert settings.postgres.password == "secret"
    assert settings.postgres.url == "postgresql+asyncpg://app:secret@postgres:5432/docs"
    assert settings.postgres.echo is True
    assert settings.postgres.pool_size == 20
    assert settings.postgres.max_overflow == 5
    assert settings.qdrant.url == "http://qdrant:6333"
    assert settings.qdrant.collection_name == "rust-docs-v2"
    assert settings.qdrant.vector_size == 768
    assert settings.qdrant.distance == "dot"
    assert settings.qdrant.upsert_batch_size == 128
    assert settings.openai.model == "gpt-5"
    assert settings.openai.api_key == "sk-test"
    assert settings.openai.max_output_tokens == 300
    assert settings.openai.temperature == 0.1
    assert settings.openai.request_timeout_seconds == 45.5
    assert settings.chat.retrieval_limit == 30
    assert settings.chat.reranking_limit == 4
    assert settings.chat.use_reranking is False
    assert settings.chat.max_query_tokens == 900
    assert settings.chat.max_context_tokens == 2400
    assert settings.embedding.model == "microsoft/harrier-oss-v1-270m"
    assert settings.embedding.base_url == "http://tei:80"
    assert settings.embedding.normalize is False
    assert settings.embedding.pooling == "mean"
    assert settings.embedding.max_batch_tokens == 8192
    assert settings.embedding.max_batch_items == 32
    assert settings.embedding.max_concurrent_requests == 12
    assert settings.embedding.request_timeout_seconds == 180.5
    assert settings.reranker.model == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker.base_url == "http://tei-reranker:80"
    assert settings.reranker.max_batch_items == 16
    assert str(settings.ingest.raw_docs_dir) == "D:\\rust-docs"
    assert settings.ingest.max_chunk_chars == 1200
    assert settings.ingest.min_chunk_chars == 120
    assert settings.logging.level == "DEBUG"
    assert settings.logging.format == "json"
    assert settings.proxy.public_base_url == "https://example.com"


def test_build_settings_rejects_invalid_boolean_values():
    try:
        build_settings({"CHAT_USE_RERANKING": "sometimes"})
    except ValueError as exc:
        assert "CHAT_USE_RERANKING" in str(exc)
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


def test_build_settings_rejects_chat_reranking_limit_above_retrieval_limit():
    with pytest.raises(ValueError, match="CHAT_RERANKING_LIMIT"):
        build_settings(
            {
                "CHAT_RETRIEVAL_LIMIT": "4",
                "CHAT_RERANKING_LIMIT": "5",
            }
        )


def test_get_settings_uses_cache_and_can_be_cleared(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("PORT", "8123")
    settings = get_settings()

    assert settings.app.port == 8123

    get_settings.cache_clear()
