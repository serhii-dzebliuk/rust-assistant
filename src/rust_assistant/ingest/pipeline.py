"""Top-level ingest pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .chunk import chunk_documents
from .chunk_dedup import deduplicate_chunks
from .clean import clean_documents
from .dedup import deduplicate_documents
from .discover import discover_documents
from .parse import parse

logger = logging.getLogger(__name__)


def run_pipeline(
    stage: str = "all",
    raw_data_dir: Path | str = "data/raw",
    crates: Optional[list[str]] = None,
    limit: Optional[int] = None,
    parsing_output: Path | str = "data/processed/docs_parsed.jsonl",
    clean_output: Path | str = "data/processed/docs_cleaned.jsonl",
    dedup_output: Path | str = "data/processed/docs_deduped.jsonl",
    chunk_output: Path | str = "data/chunks/chunks.jsonl",
    chunk_dedup_output: Path | str = "data/chunks/chunks_deduped.jsonl",
):
    """
    Execute ingest pipeline stages with a single entry point.
    Stage order: discover -> parse -> clean -> dedup -> chunk -> chunk_dedup

    Args:
        stage: Requested stage (`discover`, `parse`, `clean`, `dedup`, `chunk`, `chunk_dedup`, `all`).
        raw_data_dir: Root directory with raw HTML sources.
        crates: Optional crate filters.
        limit: Optional cap on discovered files.
        parsing_output: Output JSONL path for parsed documents.
        clean_output: Output JSONL path for cleaned documents.
        dedup_output: Output JSONL path for deduplicated documents.
        chunk_output: Output JSONL path for chunked documents.
        chunk_dedup_output: Output JSONL path for deduplicated chunks.

    Returns:
        Stage result object:
        - list[Path] for `discover`
        - list[Document] for `parse`, `clean`, and `dedup`
        - list[Chunk] for `chunk`, `chunk_dedup`, and `all`

    Example:
        >>> docs = run_pipeline(stage="all", crates=["std"], limit=100)
        >>> len(docs) > 0
        True
    """
    logger.info("Pipeline start: stage=%s", stage)
    discovered_files = discover_documents(raw_data_dir=raw_data_dir, crates=crates, limit=limit)
    logger.info("Discovery complete: %s files", len(discovered_files))
    if stage == "discover":
        return discovered_files

    parsed_docs = parse(
        html_files=discovered_files,
        raw_data_dir=raw_data_dir,
        output_file=parsing_output,
    )
    logger.info("Parse complete: %s docs", len(parsed_docs))
    if stage == "parse":
        return parsed_docs

    cleaned_docs = clean_documents(docs=parsed_docs, output_file=clean_output)
    logger.info("Clean complete: %s docs", len(cleaned_docs))
    if stage == "clean":
        return cleaned_docs

    deduped_docs = deduplicate_documents(docs=cleaned_docs, output_file=dedup_output)
    logger.info("Dedup complete: %s docs", len(deduped_docs))
    if stage == "dedup":
        return deduped_docs

    chunks = chunk_documents(docs=deduped_docs, output_file=chunk_output)
    logger.info("Chunk complete: %s chunks", len(chunks))
    if stage == "chunk":
        return chunks

    deduped_chunks = deduplicate_chunks(chunks=chunks, output_file=chunk_dedup_output)
    logger.info("Pipeline complete: %s deduplicated chunks", len(deduped_chunks))
    if stage in {"chunk_dedup", "all"}:
        return deduped_chunks

    raise ValueError(f"Unsupported stage: {stage}")
