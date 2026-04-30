from pathlib import Path

import pytest

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.bootstrap import ingest
from rust_assistant.bootstrap.settings import build_settings
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.config import SqlAlchemyConfig

pytestmark = pytest.mark.unit


def _settings(**overrides):
    env = {
        "RUST_DOCS_RAW_DIR": str(Path.cwd()),
        "DATABASE_URL": "postgresql+asyncpg://postgres:secret@postgres:5432/rust_assistant",
        "EMBEDDING_MODEL": "microsoft/harrier-oss-v1-270m",
        "EMBEDDING_BASE_URL": "http://tei:80",
        "QDRANT_URL": "http://qdrant:6333",
        "QDRANT_VECTOR_SIZE": "768",
    }
    env.update(overrides)
    return build_settings(env)


def test_build_embedding_client_requires_base_url():
    settings = _settings(EMBEDDING_BASE_URL="")

    with pytest.raises(ingest.IngestConfigurationError, match="EMBEDDING_BASE_URL"):
        ingest._build_embedding_client(settings=settings, http_client=object())


def test_build_tokenizer_requires_embedding_model():
    settings = _settings(EMBEDDING_MODEL="")

    with pytest.raises(ingest.IngestConfigurationError, match="EMBEDDING_MODEL"):
        ingest._build_tokenizer(settings)


def test_build_vector_storage_requires_qdrant_url():
    settings = _settings(QDRANT_URL="")

    with pytest.raises(ingest.IngestConfigurationError, match="QDRANT_URL"):
        ingest._build_vector_storage(settings)


def test_build_vector_storage_requires_vector_size():
    settings = _settings(QDRANT_VECTOR_SIZE="")

    with pytest.raises(ingest.IngestConfigurationError, match="QDRANT_VECTOR_SIZE"):
        ingest._build_vector_storage(settings)


def test_build_vector_storage_uses_configured_upsert_batch_size():
    settings = _settings(QDRANT_UPSERT_BATCH_SIZE="128")

    vector_storage = ingest._build_vector_storage(settings)

    assert vector_storage._upsert_batch_size == 128


def test_validate_options_rejects_limit_with_persistence_without_confirmation():
    with pytest.raises(ValueError, match="--allow-sample-persist"):
        ingest._validate_options(
            stage="all",
            persist=True,
            limit=10,
            allow_sample_persist=False,
        )


def test_validate_options_allows_limit_with_confirmed_sample_persistence():
    ingest._validate_options(
        stage="all",
        persist=True,
        limit=10,
        allow_sample_persist=True,
    )


def test_validate_options_allows_limit_without_persistence():
    ingest._validate_options(
        stage="discover",
        persist=False,
        limit=10,
        allow_sample_persist=False,
    )


@pytest.mark.asyncio
async def test_persist_after_pipeline_reports_missing_database_url():
    with pytest.raises(ingest.IngestDatabaseUnavailableError, match="DATABASE_URL"):
        await ingest._persist_after_pipeline(
            sqlalchemy_config=SqlAlchemyConfig(url=None, echo=False, pool_size=10, max_overflow=10),
            settings=_settings(),
            artifacts=IngestPipelineArtifacts(),
        )
