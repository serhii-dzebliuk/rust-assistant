import asyncio
from pathlib import Path

import pytest

from rust_assistant.core.config import build_settings, get_settings
from rust_assistant.ingest import run
from rust_assistant.ingest.persist import IngestPersistenceResult
from rust_assistant.ingest.pipeline import PipelineArtifacts

pytestmark = pytest.mark.unit


def test_build_parser_exposes_env_based_in_memory_cli_contract():
    help_text = run.build_parser().format_help()

    assert "--no-persist" in help_text
    assert "--raw-dir" not in help_text
    assert "--parse-output" not in help_text
    assert "--clean-output" not in help_text
    assert "--dedup-output" not in help_text
    assert "--chunk-output" not in help_text
    assert "--chunk-dedup-output" not in help_text
    assert "--persist-postgres" not in help_text


def test_resolve_raw_docs_dir_requires_env_value():
    parser = run.build_parser()

    with pytest.raises(SystemExit):
        run._resolve_raw_docs_dir(build_settings({}), parser)


def test_main_uses_env_raw_docs_dir_for_no_persist_run(monkeypatch):
    get_settings.cache_clear()
    raw_docs_dir = Path(".").resolve()
    captured = {}

    def fake_run_pipeline_artifacts(**kwargs):
        captured.update(kwargs)
        return PipelineArtifacts(discovered_files=[raw_docs_dir / "std/index.html"])

    async def fail_persist(**_kwargs):
        raise AssertionError("persist should not be called during --no-persist runs")

    monkeypatch.setenv("RUST_DOCS_RAW_DIR", str(raw_docs_dir))
    monkeypatch.setenv("INGEST_MAX_CHUNK_CHARS", "1234")
    monkeypatch.setenv("INGEST_MIN_CHUNK_CHARS", "123")
    monkeypatch.setattr(
        "rust_assistant.ingest.run.run_pipeline_artifacts", fake_run_pipeline_artifacts
    )
    monkeypatch.setattr("rust_assistant.ingest.run.persist_ingest_artifacts", fail_persist)

    assert run.main(["--stage", "discover", "--no-persist"]) == 0
    assert captured["raw_data_dir"] == raw_docs_dir
    assert captured["stage"] == "discover"
    assert captured["max_chunk_chars"] == 1234
    assert captured["min_chunk_chars"] == 123

    get_settings.cache_clear()


def test_main_calls_single_async_persistence_helper_after_pipeline_success(monkeypatch):
    get_settings.cache_clear()
    raw_docs_dir = Path(".").resolve()
    artifacts = PipelineArtifacts()
    calls = []

    def fake_run_pipeline_artifacts(**_kwargs):
        calls.append("pipeline")
        return artifacts

    async def fake_persist_after_pipeline(**kwargs):
        calls.append("persist")
        assert kwargs["artifacts"] is artifacts
        assert kwargs["selected_crates"] == ["std"]
        return IngestPersistenceResult(status="completed", document_count=1, chunk_count=1)

    monkeypatch.setenv("RUST_DOCS_RAW_DIR", str(raw_docs_dir))
    monkeypatch.setattr(
        "rust_assistant.ingest.run.run_pipeline_artifacts", fake_run_pipeline_artifacts
    )
    monkeypatch.setattr(
        "rust_assistant.ingest.run._persist_after_pipeline",
        fake_persist_after_pipeline,
    )

    assert run.main(["--crate", "std"]) == 0
    assert calls == ["pipeline", "persist"]

    get_settings.cache_clear()


def test_main_rejects_limited_persist_run():
    with pytest.raises(SystemExit):
        run.main(["--limit", "1"])


def test_main_does_not_persist_when_pipeline_fails(monkeypatch):
    get_settings.cache_clear()
    raw_docs_dir = Path(".").resolve()

    def fail_pipeline(**_kwargs):
        raise RuntimeError("pipeline failed")

    async def fail_persist_after_pipeline(**_kwargs):
        raise AssertionError("DB lifecycle should not start after pipeline failure")

    monkeypatch.setenv("RUST_DOCS_RAW_DIR", str(raw_docs_dir))
    monkeypatch.setattr("rust_assistant.ingest.run.run_pipeline_artifacts", fail_pipeline)
    monkeypatch.setattr(
        "rust_assistant.ingest.run._persist_after_pipeline",
        fail_persist_after_pipeline,
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        run.main([])

    get_settings.cache_clear()


def test_persist_after_pipeline_uses_one_db_lifecycle_and_disposes(monkeypatch):
    settings = build_settings({"DATABASE_URL": "postgresql+asyncpg://app:secret@db:5432/docs"})
    artifacts = PipelineArtifacts()
    events = []

    async def fake_database_is_ready(session_factory):
        events.append(("ready", session_factory))
        return True

    async def fake_persist_ingest_artifacts(**kwargs):
        events.append(("persist", kwargs["session_factory"], tuple(kwargs["replace_crates"])))
        return IngestPersistenceResult(status="completed", document_count=1, chunk_count=1)

    async def fake_dispose_engine(engine):
        events.append(("dispose", engine))

    monkeypatch.setattr("rust_assistant.ingest.run.build_async_engine", lambda _settings: "engine")
    monkeypatch.setattr(
        "rust_assistant.ingest.run.build_session_factory",
        lambda engine: f"sessions-for-{engine}",
    )
    monkeypatch.setattr("rust_assistant.ingest.run.database_is_ready", fake_database_is_ready)
    monkeypatch.setattr(
        "rust_assistant.ingest.run.persist_ingest_artifacts",
        fake_persist_ingest_artifacts,
    )
    monkeypatch.setattr("rust_assistant.ingest.run.dispose_engine", fake_dispose_engine)

    result = asyncio.run(
        run._persist_after_pipeline(
            settings=settings,
            artifacts=artifacts,
            selected_crates=["std"],
        )
    )

    assert result.status == "completed"
    assert events == [
        ("ready", "sessions-for-engine"),
        ("persist", "sessions-for-engine", ("std",)),
        ("dispose", "engine"),
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

    monkeypatch.setattr("rust_assistant.ingest.run.build_async_engine", lambda _settings: "engine")
    monkeypatch.setattr(
        "rust_assistant.ingest.run.build_session_factory", lambda _engine: "sessions"
    )
    monkeypatch.setattr("rust_assistant.ingest.run.database_is_ready", fake_database_is_ready)
    monkeypatch.setattr(
        "rust_assistant.ingest.run.persist_ingest_artifacts",
        fail_persist_ingest_artifacts,
    )
    monkeypatch.setattr("rust_assistant.ingest.run.dispose_engine", fake_dispose_engine)

    with pytest.raises(run.IngestDatabaseUnavailableError):
        asyncio.run(
            run._persist_after_pipeline(
                settings=settings,
                artifacts=PipelineArtifacts(),
                selected_crates=["std"],
            )
        )

    assert events == ["ready", "dispose"]


def test_persist_after_pipeline_disposes_when_persist_raises(monkeypatch):
    settings = build_settings({"DATABASE_URL": "postgresql+asyncpg://app:secret@db:5432/docs"})
    events = []

    async def fake_database_is_ready(_session_factory):
        events.append("ready")
        return True

    async def fail_persist_ingest_artifacts(**_kwargs):
        events.append("persist")
        raise RuntimeError("write failed")

    async def fake_dispose_engine(_engine):
        events.append("dispose")

    monkeypatch.setattr("rust_assistant.ingest.run.build_async_engine", lambda _settings: "engine")
    monkeypatch.setattr(
        "rust_assistant.ingest.run.build_session_factory", lambda _engine: "sessions"
    )
    monkeypatch.setattr("rust_assistant.ingest.run.database_is_ready", fake_database_is_ready)
    monkeypatch.setattr(
        "rust_assistant.ingest.run.persist_ingest_artifacts",
        fail_persist_ingest_artifacts,
    )
    monkeypatch.setattr("rust_assistant.ingest.run.dispose_engine", fake_dispose_engine)

    with pytest.raises(RuntimeError, match="write failed"):
        asyncio.run(
            run._persist_after_pipeline(
                settings=settings,
                artifacts=PipelineArtifacts(),
                selected_crates=["std"],
            )
        )

    assert events == ["ready", "persist", "dispose"]
