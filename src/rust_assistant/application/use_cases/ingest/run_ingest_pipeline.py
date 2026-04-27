"""Top-level ingest pipeline orchestration use case."""

from __future__ import annotations

import logging
from typing import Optional

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.discover_documents import DiscoverDocuments
from rust_assistant.application.use_cases.ingest.parse_documents import ParseDocuments
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate
from rust_assistant.domain.policies.chunk_deduplication import deduplicate_chunks
from rust_assistant.domain.policies.chunking import chunk_documents
from rust_assistant.domain.policies.document_cleaning import clean_documents
from rust_assistant.domain.policies.document_deduplication import (
    deduplicate_documents,
)

logger = logging.getLogger(__name__)


class RunIngestPipeline:
    """Execute the full ingest pipeline and return all intermediate artifacts."""

    def __init__(
        self,
        *,
        discover_documents: DiscoverDocuments,
        parse_documents: ParseDocuments,
    ):
        self._discover_documents = discover_documents
        self._parse_documents = parse_documents

    def execute(
        self,
        *,
        stage: str = "all",
        crates: Optional[list[Crate]] = None,
        limit: Optional[int] = None,
        max_chunk_chars: int = 1400,
        min_chunk_chars: int = 180,
    ) -> IngestPipelineArtifacts:
        """Execute the ingest pipeline and return all intermediate artifacts."""
        logger.info("Pipeline start: stage=%s", stage)
        discovered_files = self._discover_documents.execute(crates=crates, limit=limit)
        logger.info("Discovery complete: %s files", len(discovered_files))
        if stage == "discover":
            return IngestPipelineArtifacts(discovered_files=discovered_files)

        parsed_docs = self._parse_documents.execute(html_files=discovered_files)
        logger.info("Parse complete: %s docs", len(parsed_docs))
        if stage == "parse":
            return IngestPipelineArtifacts(discovered_files=discovered_files, parsed_docs=parsed_docs)

        cleaned_docs = clean_documents(documents=parsed_docs)
        logger.info("Clean complete: %s docs", len(cleaned_docs))
        if stage == "clean":
            return IngestPipelineArtifacts(
                discovered_files=discovered_files,
                parsed_docs=parsed_docs,
                cleaned_docs=cleaned_docs,
            )

        deduped_docs = deduplicate_documents(documents=cleaned_docs)
        logger.info("Dedup complete: %s docs", len(deduped_docs))
        if stage == "dedup":
            return IngestPipelineArtifacts(
                discovered_files=discovered_files,
                parsed_docs=parsed_docs,
                cleaned_docs=cleaned_docs,
                deduped_docs=deduped_docs,
            )

        chunks = chunk_documents(
            documents=deduped_docs,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )
        logger.info("Chunk complete: %s chunks", len(chunks))
        if stage == "chunk":
            return IngestPipelineArtifacts(
                discovered_files=discovered_files,
                parsed_docs=parsed_docs,
                cleaned_docs=cleaned_docs,
                deduped_docs=deduped_docs,
                chunks=chunks,
            )

        deduped_chunks = deduplicate_chunks(chunks=chunks, documents=deduped_docs)
        logger.info("Pipeline complete: %s deduplicated chunks", len(deduped_chunks))
        if stage in {"chunk_dedup", "all"}:
            return IngestPipelineArtifacts(
                discovered_files=discovered_files,
                parsed_docs=parsed_docs,
                cleaned_docs=cleaned_docs,
                deduped_docs=deduped_docs,
                chunks=chunks,
                deduped_chunks=deduped_chunks,
            )

        raise ValueError(f"Unsupported stage: {stage}")
