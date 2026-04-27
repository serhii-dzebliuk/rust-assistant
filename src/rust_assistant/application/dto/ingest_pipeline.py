"""Application DTOs for ingest pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document


@dataclass(slots=True, frozen=True)
class IngestPipelineArtifacts:
    """All intermediate ingest outputs produced by a pipeline run."""

    discovered_files: list[Path] = field(default_factory=list)
    parsed_docs: list[Document] = field(default_factory=list)
    cleaned_docs: list[Document] = field(default_factory=list)
    deduped_docs: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    deduped_chunks: list[Chunk] = field(default_factory=list)
