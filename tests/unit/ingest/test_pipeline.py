from pathlib import Path

import pytest

from rust_assistant.ingest.pipeline import PipelineArtifacts, run_pipeline, run_pipeline_artifacts

pytestmark = pytest.mark.unit


def test_run_pipeline_artifacts_returns_full_stage_outputs(monkeypatch):
    discovered = [Path("data/raw/std/index.html")]
    parsed = ["parsed"]
    cleaned = ["cleaned"]
    deduped = ["deduped"]
    chunks = ["chunks"]
    deduped_chunks = ["deduped_chunks"]

    monkeypatch.setattr("rust_assistant.ingest.pipeline.discover_documents", lambda **_: discovered)
    monkeypatch.setattr("rust_assistant.ingest.pipeline.parse", lambda **_: parsed)
    monkeypatch.setattr("rust_assistant.ingest.pipeline.clean_documents", lambda **_: cleaned)
    monkeypatch.setattr("rust_assistant.ingest.pipeline.deduplicate_documents", lambda **_: deduped)
    monkeypatch.setattr("rust_assistant.ingest.pipeline.chunk_documents", lambda **_: chunks)
    monkeypatch.setattr("rust_assistant.ingest.pipeline.deduplicate_chunks", lambda **_: deduped_chunks)

    artifacts = run_pipeline_artifacts(stage="all")

    assert artifacts == PipelineArtifacts(
        discovered_files=discovered,
        parsed_docs=parsed,
        cleaned_docs=cleaned,
        deduped_docs=deduped,
        chunks=chunks,
        deduped_chunks=deduped_chunks,
    )


def test_run_pipeline_keeps_historical_stage_specific_return_values(monkeypatch):
    parsed = ["parsed"]

    monkeypatch.setattr(
        "rust_assistant.ingest.pipeline.run_pipeline_artifacts",
        lambda **_: PipelineArtifacts(parsed_docs=parsed),
    )

    assert run_pipeline(stage="parse") == parsed
