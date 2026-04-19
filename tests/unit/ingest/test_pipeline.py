from pathlib import Path

import pytest

from rust_assistant.ingest.pipeline import PipelineArtifacts, run_pipeline, run_pipeline_artifacts

pytestmark = pytest.mark.unit


def test_run_pipeline_artifacts_returns_full_stage_outputs(monkeypatch):
    raw_docs_dir = Path("rust-docs")
    discovered = [Path("rust-docs/std/index.html")]
    parsed = ["parsed"]
    cleaned = ["cleaned"]
    deduped = ["deduped"]
    chunks = ["chunks"]
    deduped_chunks = ["deduped_chunks"]

    def fake_parse(**kwargs):
        assert "output_file" not in kwargs
        return parsed

    def fake_clean_documents(**kwargs):
        assert "output_file" not in kwargs
        return cleaned

    def fake_deduplicate_documents(**kwargs):
        assert "output_file" not in kwargs
        return deduped

    def fake_chunk_documents(**kwargs):
        assert "output_file" not in kwargs
        assert kwargs["max_chunk_chars"] == 1200
        assert kwargs["min_chunk_chars"] == 120
        return chunks

    def fake_deduplicate_chunks(**kwargs):
        assert "output_file" not in kwargs
        return deduped_chunks

    monkeypatch.setattr("rust_assistant.ingest.pipeline.discover_documents", lambda **_: discovered)
    monkeypatch.setattr("rust_assistant.ingest.pipeline.parse", fake_parse)
    monkeypatch.setattr("rust_assistant.ingest.pipeline.clean_documents", fake_clean_documents)
    monkeypatch.setattr(
        "rust_assistant.ingest.pipeline.deduplicate_documents",
        fake_deduplicate_documents,
    )
    monkeypatch.setattr("rust_assistant.ingest.pipeline.chunk_documents", fake_chunk_documents)
    monkeypatch.setattr(
        "rust_assistant.ingest.pipeline.deduplicate_chunks",
        fake_deduplicate_chunks,
    )

    artifacts = run_pipeline_artifacts(
        raw_data_dir=raw_docs_dir,
        stage="all",
        max_chunk_chars=1200,
        min_chunk_chars=120,
    )

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

    assert run_pipeline(raw_data_dir=Path("rust-docs"), stage="parse") == parsed
