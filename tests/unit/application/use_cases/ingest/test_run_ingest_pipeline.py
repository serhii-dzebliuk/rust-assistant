from pathlib import Path

import pytest

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.run_ingest_pipeline import (
    RunIngestPipeline,
)

pytestmark = pytest.mark.unit


class FakeDiscoverDocuments:
    def __init__(self, discovered_files):
        self.discovered_files = discovered_files

    def execute(self, *, crates=None, limit=None):
        return self.discovered_files


class FakeParseDocuments:
    def __init__(self, parsed_docs):
        self.parsed_docs = parsed_docs

    def execute(self, *, html_files):
        return self.parsed_docs


def test_run_ingest_pipeline_returns_full_stage_outputs(monkeypatch):
    discovered = [Path("rust-docs/std/index.html")]
    parsed = ["parsed"]
    cleaned = ["cleaned"]
    deduped = ["deduped"]
    chunks = ["chunks"]
    deduped_chunks = ["deduped_chunks"]

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

    monkeypatch.setattr(
        "rust_assistant.application.use_cases.ingest.run_ingest_pipeline.clean_documents",
        fake_clean_documents,
    )
    monkeypatch.setattr(
        "rust_assistant.application.use_cases.ingest.run_ingest_pipeline.deduplicate_documents",
        fake_deduplicate_documents,
    )
    monkeypatch.setattr(
        "rust_assistant.application.use_cases.ingest.run_ingest_pipeline.chunk_documents",
        fake_chunk_documents,
    )
    monkeypatch.setattr(
        "rust_assistant.application.use_cases.ingest.run_ingest_pipeline.deduplicate_chunks",
        fake_deduplicate_chunks,
    )

    pipeline = RunIngestPipeline(
        discover_documents=FakeDiscoverDocuments(discovered),
        parse_documents=FakeParseDocuments(parsed),
    )
    artifacts = pipeline.execute(
        stage="all",
        max_chunk_chars=1200,
        min_chunk_chars=120,
    )

    assert artifacts == IngestPipelineArtifacts(
        discovered_files=discovered,
        parsed_docs=parsed,
        cleaned_docs=cleaned,
        deduped_docs=deduped,
        chunks=chunks,
        deduped_chunks=deduped_chunks,
    )


def test_run_ingest_pipeline_returns_requested_stage_artifacts_only():
    discovered = [Path("rust-docs/std/index.html")]
    parsed = ["parsed"]

    pipeline = RunIngestPipeline(
        discover_documents=FakeDiscoverDocuments(discovered),
        parse_documents=FakeParseDocuments(parsed),
    )

    artifacts = pipeline.execute(stage="parse")

    assert artifacts == IngestPipelineArtifacts(
        discovered_files=discovered,
        parsed_docs=parsed,
    )
