from pathlib import Path

import pytest

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.discover_documents import (
    DiscoverDocumentsResult,
)
from rust_assistant.application.use_cases.ingest.ingest_documents import (
    IngestDocumentsCommand,
    IngestDocumentsUseCase,
)
from rust_assistant.application.use_cases.ingest.parse_documents import ParseDocumentsResult

pytestmark = pytest.mark.unit


class FakeDiscoverDocumentsUseCase:
    def __init__(self, discovered_files):
        self.discovered_files = discovered_files
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return DiscoverDocumentsResult(discovered_files=self.discovered_files)


class FakeParseDocumentsUseCase:
    def __init__(self, parsed_docs, failures=None):
        self.parsed_docs = parsed_docs
        self.failures = failures or []
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return ParseDocumentsResult(documents=self.parsed_docs, failures=self.failures)


def test_ingest_documents_returns_full_stage_outputs(monkeypatch):
    discovered = [Path("rust-docs/std/index.html")]
    parsed = ["parsed"]
    parse_failures = ["parse_failure"]
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
        "rust_assistant.application.use_cases.ingest.ingest_documents.clean_documents",
        fake_clean_documents,
    )
    monkeypatch.setattr(
        "rust_assistant.application.use_cases.ingest.ingest_documents.deduplicate_documents",
        fake_deduplicate_documents,
    )
    monkeypatch.setattr(
        "rust_assistant.application.use_cases.ingest.ingest_documents.chunk_documents",
        fake_chunk_documents,
    )
    monkeypatch.setattr(
        "rust_assistant.application.use_cases.ingest.ingest_documents.deduplicate_chunks",
        fake_deduplicate_chunks,
    )

    use_case = IngestDocumentsUseCase(
        discover_documents=FakeDiscoverDocumentsUseCase(discovered),
        parse_documents=FakeParseDocumentsUseCase(parsed, parse_failures),
    )
    result = use_case.execute(
        IngestDocumentsCommand(
            stage="all",
            max_chunk_chars=1200,
            min_chunk_chars=120,
        )
    )

    assert result.artifacts == IngestPipelineArtifacts(
        discovered_files=discovered,
        parsed_docs=parsed,
        parse_failures=parse_failures,
        cleaned_docs=cleaned,
        deduped_docs=deduped,
        chunks=chunks,
        deduped_chunks=deduped_chunks,
    )


def test_ingest_documents_returns_requested_stage_artifacts_only():
    discovered = [Path("rust-docs/std/index.html")]
    parsed = ["parsed"]
    parse_failures = ["parse_failure"]

    use_case = IngestDocumentsUseCase(
        discover_documents=FakeDiscoverDocumentsUseCase(discovered),
        parse_documents=FakeParseDocumentsUseCase(parsed, parse_failures),
    )

    result = use_case.execute(IngestDocumentsCommand(stage="parse"))

    assert result.artifacts == IngestPipelineArtifacts(
        discovered_files=discovered,
        parsed_docs=parsed,
        parse_failures=parse_failures,
    )
