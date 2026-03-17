"""Top-level ingest pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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
):
    """
    Execute ingest pipeline stages with a single entry point.
    Stage order: discover -> parse -> clean -> dedup

    Args:
        stage: Requested stage (`discover`, `parse`, `clean`, `dedup`, `all`).
        raw_data_dir: Root directory with raw HTML sources.
        crates: Optional crate filters.
        limit: Optional cap on discovered files.
        parsing_output: Output JSONL path for parsed documents.
        clean_output: Output JSONL path for cleaned documents.
        dedup_output: Output JSONL path for deduplicated documents.

    Returns:
        Stage result object:
        - list[Path] for `discover`
        - list[Document] for other stages

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
    logger.info("Pipeline complete: %s deduplicated docs", len(deduped_docs))
    if stage in {"dedup", "all"}:
        return deduped_docs

    raise ValueError(f"Unsupported stage: {stage}")
