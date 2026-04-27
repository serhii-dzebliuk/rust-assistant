import asyncio
from pathlib import Path

import pytest

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.rebuild_knowledge_base import (
    RebuildKnowledgeBaseResult,
)
from rust_assistant.domain.enums import Crate
from rust_assistant.infrastructure.outbound.sqlalchemy.config import SqlAlchemyConfig
from rust_assistant.bootstrap import ingest
from rust_assistant.bootstrap.settings import build_settings, get_settings

pytestmark = pytest.mark.unit


def _sqlalchemy_config(url: str = "postgresql+asyncpg://app:secret@db:5432/docs") -> SqlAlchemyConfig:
    return SqlAlchemyConfig(url=url, echo=False, pool_size=10, max_overflow=10)


def test_resolve_raw_docs_dir_requires_env_value():
    with pytest.raises(ValueError, match="RUST_DOCS_RAW_DIR"):
        ingest._resolve_raw_docs_dir(build_settings({}))


def test_run_ingest_uses_env_raw_docs_dir_for_no_persist_run(monkeypatch):
    get_settings.cache_clear()
    raw_docs_dir = Path(".").resolve()
    captured = {}

    def fake_run_pipeline_artifacts(**kwargs):
        captured.update(kwargs)
        return IngestPipelineArtifacts(discovered_files=[raw_docs_dir / "std/index.html"])

    monkeypatch.setenv("RUST_DOCS_RAW_DIR", str(raw_docs_dir))
    monkeypatch.setenv("INGEST_MAX_CHUNK_CHARS", "1234")
    monkeypatch.setenv("INGEST_MIN_CHUNK_CHARS", "123")
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest._run_pipeline_artifacts",
        fake_run_pipeline_artifacts,
    )

    assert ingest.run_ingest(stage="discover", persist=False) == 0
    assert captured["raw_docs_dir"] == raw_docs_dir
    assert captured["stage"] == "discover"
    assert captured["crates"] == [Crate.STD, Crate.BOOK, Crate.CARGO, Crate.REFERENCE]
    assert captured["max_chunk_chars"] == 1234
    assert captured["min_chunk_chars"] == 123

    get_settings.cache_clear()


def test_run_ingest_calls_single_async_persistence_helper_after_pipeline_success(monkeypatch):
    get_settings.cache_clear()
    raw_docs_dir = Path(".").resolve()
    artifacts = IngestPipelineArtifacts()
    calls = []

    def fake_run_pipeline_artifacts(**_kwargs):
        calls.append("pipeline")
        return artifacts

    async def fake_persist_after_pipeline(**kwargs):
        calls.append("persist")
        assert kwargs["artifacts"] is artifacts
        return RebuildKnowledgeBaseResult(status="completed", document_count=1, chunk_count=1)

    monkeypatch.setenv("RUST_DOCS_RAW_DIR", str(raw_docs_dir))
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest._run_pipeline_artifacts",
        fake_run_pipeline_artifacts,
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest._persist_after_pipeline",
        fake_persist_after_pipeline,
    )

    assert ingest.run_ingest(crates=["std"]) == 0
    assert calls == ["pipeline", "persist"]

    get_settings.cache_clear()


def test_run_ingest_rejects_limited_persist_run():
    with pytest.raises(ValueError, match="--limit is only allowed"):
        ingest.run_ingest(limit=1)


def test_run_ingest_does_not_persist_when_pipeline_fails(monkeypatch):
    get_settings.cache_clear()
    raw_docs_dir = Path(".").resolve()

    def fail_pipeline(**_kwargs):
        raise RuntimeError("pipeline failed")

    async def fail_persist_after_pipeline(**_kwargs):
        raise AssertionError("DB lifecycle should not start after pipeline failure")

    monkeypatch.setenv("RUST_DOCS_RAW_DIR", str(raw_docs_dir))
    monkeypatch.setattr("rust_assistant.bootstrap.ingest._run_pipeline_artifacts", fail_pipeline)
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest._persist_after_pipeline",
        fail_persist_after_pipeline,
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        ingest.run_ingest()

    get_settings.cache_clear()


def test_persist_after_pipeline_uses_one_db_lifecycle_and_disposes(monkeypatch):
    settings = build_settings({"DATABASE_URL": "postgresql+asyncpg://app:secret@db:5432/docs"})
    artifacts = IngestPipelineArtifacts()
    events = []

    async def fake_database_is_ready(session_factory):
        events.append(("ready", session_factory))
        return True

    class FakeRebuildKnowledgeBase:
        async def execute(self, *, artifacts, uow, token_counter):
            events.append(("persist", uow))
            assert artifacts is not None
            assert token_counter is None
            return RebuildKnowledgeBaseResult(status="completed", document_count=1, chunk_count=1)

    async def fake_dispose_engine(engine):
        events.append(("dispose", engine))

    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.build_async_engine", lambda _config: "engine"
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.build_session_factory",
        lambda engine: f"sessions-for-{engine}",
    )
    monkeypatch.setattr("rust_assistant.bootstrap.ingest.database_is_ready", fake_database_is_ready)
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.SqlAlchemyUnitOfWork",
        lambda session_factory: f"uow-for-{session_factory}",
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.RebuildKnowledgeBase",
        FakeRebuildKnowledgeBase,
    )
    monkeypatch.setattr("rust_assistant.bootstrap.ingest.dispose_engine", fake_dispose_engine)

    result = asyncio.run(
        ingest._persist_after_pipeline(
            sqlalchemy_config=_sqlalchemy_config(),
            settings=settings,
            artifacts=artifacts,
        )
    )

    assert result.status == "completed"
    assert events == [
        ("ready", "sessions-for-engine"),
        ("persist", "uow-for-sessions-for-engine"),
        ("dispose", "engine"),
    ]


def test_persist_after_pipeline_uses_embedding_model_token_counter(monkeypatch):
    settings = build_settings(
        {
            "DATABASE_URL": "postgresql+asyncpg://app:secret@db:5432/docs",
            "EMBEDDING_MODEL": "microsoft/harrier-oss-v1-270m",
        }
    )
    artifacts = IngestPipelineArtifacts()
    events = []

    class FakeChunkTokenCounter:
        @classmethod
        def from_model_name(cls, model_name):
            events.append(("tokenizer", model_name))
            return "token-counter"

    async def fake_database_is_ready(_session_factory):
        events.append("ready")
        return True

    class FakeRebuildKnowledgeBase:
        async def execute(self, *, artifacts, uow, token_counter):
            events.append(("persist", token_counter))
            return RebuildKnowledgeBaseResult(status="completed", document_count=1, chunk_count=1)

    async def fake_dispose_engine(_engine):
        events.append("dispose")

    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.build_async_engine", lambda _config: "engine"
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.build_session_factory",
        lambda _engine: "sessions",
    )
    monkeypatch.setattr("rust_assistant.bootstrap.ingest.database_is_ready", fake_database_is_ready)
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.SqlAlchemyUnitOfWork",
        lambda _session_factory: "uow",
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.RebuildKnowledgeBase",
        FakeRebuildKnowledgeBase,
    )
    monkeypatch.setattr("rust_assistant.bootstrap.ingest.dispose_engine", fake_dispose_engine)
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.TransformersChunkTokenCounter",
        FakeChunkTokenCounter,
    )

    result = asyncio.run(
        ingest._persist_after_pipeline(
            sqlalchemy_config=_sqlalchemy_config(),
            settings=settings,
            artifacts=artifacts,
        )
    )

    assert result.status == "completed"
    assert events == [
        "ready",
        ("tokenizer", "microsoft/harrier-oss-v1-270m"),
        ("persist", "token-counter"),
        "dispose",
    ]


def test_persist_after_pipeline_disposes_when_database_is_not_ready(monkeypatch):
    settings = build_settings({"DATABASE_URL": "postgresql+asyncpg://app:secret@db:5432/docs"})
    events = []

    async def fake_database_is_ready(_session_factory):
        events.append("ready")
        return False

    async def fail_persist_ingest_artifacts(**_kwargs):
        raise AssertionError("persist should not run when database is not ready")

    async def fake_dispose_engine(_engine):
        events.append("dispose")

    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.build_async_engine", lambda _config: "engine"
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.build_session_factory", lambda _engine: "sessions"
    )
    monkeypatch.setattr("rust_assistant.bootstrap.ingest.database_is_ready", fake_database_is_ready)
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.RebuildKnowledgeBase",
        lambda: None,
    )
    monkeypatch.setattr("rust_assistant.bootstrap.ingest.dispose_engine", fake_dispose_engine)

    with pytest.raises(ingest.IngestDatabaseUnavailableError):
        asyncio.run(
            ingest._persist_after_pipeline(
                sqlalchemy_config=_sqlalchemy_config(),
                settings=settings,
                artifacts=IngestPipelineArtifacts(),
            )
        )

    assert events == ["ready", "dispose"]


def test_persist_after_pipeline_disposes_when_persist_raises(monkeypatch):
    settings = build_settings({"DATABASE_URL": "postgresql+asyncpg://app:secret@db:5432/docs"})
    events = []

    async def fake_database_is_ready(_session_factory):
        events.append("ready")
        return True

    class FailRebuildKnowledgeBase:
        async def execute(self, *, artifacts, uow, token_counter):
            events.append("persist")
            raise RuntimeError("write failed")

    async def fake_dispose_engine(_engine):
        events.append("dispose")

    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.build_async_engine", lambda _config: "engine"
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.build_session_factory", lambda _engine: "sessions"
    )
    monkeypatch.setattr("rust_assistant.bootstrap.ingest.database_is_ready", fake_database_is_ready)
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.SqlAlchemyUnitOfWork",
        lambda _session_factory: "uow",
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.RebuildKnowledgeBase",
        FailRebuildKnowledgeBase,
    )
    monkeypatch.setattr("rust_assistant.bootstrap.ingest.dispose_engine", fake_dispose_engine)

    with pytest.raises(RuntimeError, match="write failed"):
        asyncio.run(
            ingest._persist_after_pipeline(
                sqlalchemy_config=_sqlalchemy_config(),
                settings=settings,
                artifacts=IngestPipelineArtifacts(),
            )
        )

    assert events == ["ready", "persist", "dispose"]
