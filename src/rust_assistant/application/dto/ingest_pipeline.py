"""Application DTOs for ingest pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rust_assistant.application.dto.document_parse import DocumentParseFailure
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document


@dataclass(slots=True, frozen=True)
class IngestPipelineArtifacts:
    """All intermediate ingest outputs produced by a pipeline run."""

    discovered_files: list[Path] = field(default_factory=list[Path])
    parsed_docs: list[Document] = field(default_factory=list[Document])
    parse_failures: list[DocumentParseFailure] = field(default_factory=list[DocumentParseFailure])
    cleaned_docs: list[Document] = field(default_factory=list[Document])
    deduped_docs: list[Document] = field(default_factory=list[Document])
    chunks: list[Chunk] = field(default_factory=list[Chunk])
    deduped_chunks: list[Chunk] = field(default_factory=list[Chunk])
