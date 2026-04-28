"""Top-level ingest pipeline orchestration use case."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.discover_documents import (
    DiscoverDocumentsCommand,
    DiscoverDocumentsUseCase,
)
from rust_assistant.application.use_cases.ingest.parse_documents import (
    ParseDocumentsCommand,
    ParseDocumentsUseCase,
)
from rust_assistant.domain.enums import Crate
from rust_assistant.domain.policies.chunk_deduplication import deduplicate_chunks
from rust_assistant.domain.policies.chunking import chunk_documents
from rust_assistant.domain.policies.document_cleaning import clean_documents
from rust_assistant.domain.policies.document_deduplication import (
    deduplicate_documents,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class IngestDocumentsCommand:
    """Input for executing the ingest document pipeline."""

    stage: str = "all"
    crates: Optional[list[Crate]] = None
    limit: Optional[int] = None
    max_chunk_chars: int = 1400
    min_chunk_chars: int = 180


@dataclass(slots=True, frozen=True)
class IngestDocumentsResult:
    """Output of the ingest document pipeline."""

    artifacts: IngestPipelineArtifacts


class IngestDocumentsUseCase:
    """Execute the full ingest pipeline and return all intermediate artifacts."""

    def __init__(
        self,
        *,
        discover_documents: DiscoverDocumentsUseCase,
        parse_documents: ParseDocumentsUseCase,
    ):
        self._discover_documents = discover_documents
        self._parse_documents = parse_documents

    def execute(self, command: IngestDocumentsCommand) -> IngestDocumentsResult:
        """Execute the ingest pipeline and return all intermediate artifacts."""
        stage = command.stage
        logger.info("Pipeline start: stage=%s", stage)
        discovery_result = self._discover_documents.execute(
            DiscoverDocumentsCommand(crates=command.crates, limit=command.limit)
        )
        discovered_files = discovery_result.discovered_files
        logger.info("Discovery complete: %s files", len(discovered_files))
        if stage == "discover":
            return IngestDocumentsResult(
                artifacts=IngestPipelineArtifacts(discovered_files=discovered_files)
            )

        parse_result = self._parse_documents.execute(
            ParseDocumentsCommand(html_files=discovered_files)
        )
        parsed_docs = parse_result.documents
        parse_failures = parse_result.failures
        logger.info("Parse complete: %s docs", len(parsed_docs))
        if stage == "parse":
            return IngestDocumentsResult(
                artifacts=IngestPipelineArtifacts(
                    discovered_files=discovered_files,
                    parsed_docs=parsed_docs,
                    parse_failures=parse_failures,
                )
            )

        cleaned_docs = clean_documents(documents=parsed_docs)
        logger.info("Clean complete: %s docs", len(cleaned_docs))
        if stage == "clean":
            return IngestDocumentsResult(
                artifacts=IngestPipelineArtifacts(
                    discovered_files=discovered_files,
                    parsed_docs=parsed_docs,
                    parse_failures=parse_failures,
                    cleaned_docs=cleaned_docs,
                )
            )

        deduped_docs = deduplicate_documents(documents=cleaned_docs)
        logger.info("Dedup complete: %s docs", len(deduped_docs))
        if stage == "dedup":
            return IngestDocumentsResult(
                artifacts=IngestPipelineArtifacts(
                    discovered_files=discovered_files,
                    parsed_docs=parsed_docs,
                    parse_failures=parse_failures,
                    cleaned_docs=cleaned_docs,
                    deduped_docs=deduped_docs,
                )
            )

        chunks = chunk_documents(
            documents=deduped_docs,
            max_chunk_chars=command.max_chunk_chars,
            min_chunk_chars=command.min_chunk_chars,
        )
        logger.info("Chunk complete: %s chunks", len(chunks))
        if stage == "chunk":
            return IngestDocumentsResult(
                artifacts=IngestPipelineArtifacts(
                    discovered_files=discovered_files,
                    parsed_docs=parsed_docs,
                    parse_failures=parse_failures,
                    cleaned_docs=cleaned_docs,
                    deduped_docs=deduped_docs,
                    chunks=chunks,
                )
            )

        deduped_chunks = deduplicate_chunks(chunks=chunks, documents=deduped_docs)
        logger.info("Pipeline complete: %s deduplicated chunks", len(deduped_chunks))
        if stage in {"chunk_dedup", "all"}:
            return IngestDocumentsResult(
                artifacts=IngestPipelineArtifacts(
                    discovered_files=discovered_files,
                    parsed_docs=parsed_docs,
                    parse_failures=parse_failures,
                    cleaned_docs=cleaned_docs,
                    deduped_docs=deduped_docs,
                    chunks=chunks,
                    deduped_chunks=deduped_chunks,
                )
            )

        raise ValueError(f"Unsupported stage: {stage}")
