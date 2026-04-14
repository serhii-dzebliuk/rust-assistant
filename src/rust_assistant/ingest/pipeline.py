"""Top-level ingest pipeline orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from rust_assistant.ingest.entities import Chunk, Document

from .chunk import chunk_documents
from .chunk_dedup import deduplicate_chunks
from .clean import clean_documents
from .dedup import deduplicate_documents
from .discover import discover_documents
from .parse import parse

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PipelineArtifacts:
    """All intermediate ingest outputs produced by a pipeline run."""

    discovered_files: list[Path] = field(default_factory=list)
    parsed_docs: list[Document] = field(default_factory=list)
    cleaned_docs: list[Document] = field(default_factory=list)
    deduped_docs: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    deduped_chunks: list[Chunk] = field(default_factory=list)


def run_pipeline_artifacts(
    stage: str = "all",
    raw_data_dir: Union[Path, str] = "data/raw",
    crates: Optional[list[str]] = None,
    limit: Optional[int] = None,
    parsing_output: Union[Path, str] = "data/processed/docs_parsed.jsonl",
    clean_output: Union[Path, str] = "data/processed/docs_cleaned.jsonl",
    dedup_output: Union[Path, str] = "data/processed/docs_deduped.jsonl",
    chunk_output: Union[Path, str] = "data/chunks/chunks.jsonl",
    chunk_dedup_output: Union[Path, str] = "data/chunks/chunks_deduped.jsonl",
) -> PipelineArtifacts:
    """Execute the ingest pipeline and return all intermediate artifacts."""
    logger.info("Pipeline start: stage=%s", stage)
    discovered_files = discover_documents(raw_data_dir=raw_data_dir, crates=crates, limit=limit)
    logger.info("Discovery complete: %s files", len(discovered_files))
    if stage == "discover":
        return PipelineArtifacts(discovered_files=discovered_files)

    parsed_docs = parse(
        html_files=discovered_files,
        raw_data_dir=raw_data_dir,
        output_file=parsing_output,
    )
    logger.info("Parse complete: %s docs", len(parsed_docs))
    if stage == "parse":
        return PipelineArtifacts(discovered_files=discovered_files, parsed_docs=parsed_docs)

    cleaned_docs = clean_documents(docs=parsed_docs, output_file=clean_output)
    logger.info("Clean complete: %s docs", len(cleaned_docs))
    if stage == "clean":
        return PipelineArtifacts(
            discovered_files=discovered_files,
            parsed_docs=parsed_docs,
            cleaned_docs=cleaned_docs,
        )

    deduped_docs = deduplicate_documents(docs=cleaned_docs, output_file=dedup_output)
    logger.info("Dedup complete: %s docs", len(deduped_docs))
    if stage == "dedup":
        return PipelineArtifacts(
            discovered_files=discovered_files,
            parsed_docs=parsed_docs,
            cleaned_docs=cleaned_docs,
            deduped_docs=deduped_docs,
        )

    chunks = chunk_documents(docs=deduped_docs, output_file=chunk_output)
    logger.info("Chunk complete: %s chunks", len(chunks))
    if stage == "chunk":
        return PipelineArtifacts(
            discovered_files=discovered_files,
            parsed_docs=parsed_docs,
            cleaned_docs=cleaned_docs,
            deduped_docs=deduped_docs,
            chunks=chunks,
        )

    deduped_chunks = deduplicate_chunks(chunks=chunks, output_file=chunk_dedup_output)
    logger.info("Pipeline complete: %s deduplicated chunks", len(deduped_chunks))
    if stage in {"chunk_dedup", "all"}:
        return PipelineArtifacts(
            discovered_files=discovered_files,
            parsed_docs=parsed_docs,
            cleaned_docs=cleaned_docs,
            deduped_docs=deduped_docs,
            chunks=chunks,
            deduped_chunks=deduped_chunks,
        )

    raise ValueError(f"Unsupported stage: {stage}")


def run_pipeline(
    stage: str = "all",
    raw_data_dir: Union[Path, str] = "data/raw",
    crates: Optional[list[str]] = None,
    limit: Optional[int] = None,
    parsing_output: Union[Path, str] = "data/processed/docs_parsed.jsonl",
    clean_output: Union[Path, str] = "data/processed/docs_cleaned.jsonl",
    dedup_output: Union[Path, str] = "data/processed/docs_deduped.jsonl",
    chunk_output: Union[Path, str] = "data/chunks/chunks.jsonl",
    chunk_dedup_output: Union[Path, str] = "data/chunks/chunks_deduped.jsonl",
):
    """Execute ingest pipeline stages with the historical stage-specific return values."""
    artifacts = run_pipeline_artifacts(
        stage=stage,
        raw_data_dir=raw_data_dir,
        crates=crates,
        limit=limit,
        parsing_output=parsing_output,
        clean_output=clean_output,
        dedup_output=dedup_output,
        chunk_output=chunk_output,
        chunk_dedup_output=chunk_dedup_output,
    )

    if stage == "discover":
        return artifacts.discovered_files
    if stage == "parse":
        return artifacts.parsed_docs
    if stage == "clean":
        return artifacts.cleaned_docs
    if stage == "dedup":
        return artifacts.deduped_docs
    if stage == "chunk":
        return artifacts.chunks
    if stage in {"chunk_dedup", "all"}:
        return artifacts.deduped_chunks
    raise ValueError(f"Unsupported stage: {stage}")
