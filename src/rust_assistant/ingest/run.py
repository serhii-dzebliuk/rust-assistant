"""CLI entry point for ingest pipeline execution."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Optional

from rust_assistant.core.config import Settings, get_settings
from rust_assistant.core.db import (
    build_async_engine,
    build_session_factory,
    database_is_ready,
    dispose_engine,
)
from rust_assistant.core.logging import configure_logging

from .persist import IngestPersistenceResult, persist_ingest_artifacts
from .pipeline import PipelineArtifacts, run_pipeline_artifacts
from .token_count import ChunkTokenCounter

logger = logging.getLogger(__name__)

SUPPORTED_CRATES = ("std", "book", "cargo", "reference")
PERSISTABLE_STAGES = ("chunk_dedup", "all")


class IngestDatabaseUnavailableError(RuntimeError):
    """Raised when the ingest CLI cannot reach PostgreSQL."""


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for ingest pipeline execution."""
    parser = argparse.ArgumentParser(description="Run the rust-assistant ingest pipeline")
    parser.add_argument(
        "--stage",
        choices=["discover", "parse", "clean", "dedup", "chunk", "chunk_dedup", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--crate",
        type=str,
        action="append",
        choices=SUPPORTED_CRATES,
        help="Crate(s) to include (can be specified multiple times, default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of files to process. Requires --no-persist.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Run the pipeline without replacing PostgreSQL ingest data",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def _resolve_raw_docs_dir(settings: Settings, parser: argparse.ArgumentParser) -> Path:
    """Resolve and validate the raw Rust documentation directory for ingest."""
    raw_docs_dir = settings.ingest.raw_docs_dir
    if raw_docs_dir is None:
        parser.error("RUST_DOCS_RAW_DIR must be configured before running ingest")

    resolved = raw_docs_dir.expanduser().resolve()
    if not resolved.exists():
        parser.error(f"RUST_DOCS_RAW_DIR does not exist: {resolved}")
    if not resolved.is_dir():
        parser.error(f"RUST_DOCS_RAW_DIR must point to a directory: {resolved}")
    return resolved


def _selected_crates(args: argparse.Namespace) -> list[str]:
    """Return the crate scope selected for this ingest run."""
    return list(dict.fromkeys(args.crate or SUPPORTED_CRATES))


def _log_stage_summary(stage: str, artifacts: PipelineArtifacts) -> None:
    """Log a concise summary for the completed ingest stage."""
    logger.info("Discovered files: %s", len(artifacts.discovered_files))
    if stage == "discover":
        return

    logger.info("Parsed documents: %s", len(artifacts.parsed_docs))
    if stage == "parse":
        return

    logger.info("Cleaned documents: %s", len(artifacts.cleaned_docs))
    if stage == "clean":
        return

    logger.info("Deduplicated documents: %s", len(artifacts.deduped_docs))
    if stage == "dedup":
        return

    logger.info("Generated chunks: %s", len(artifacts.chunks))
    if stage == "chunk":
        return

    logger.info("Deduplicated chunks: %s", len(artifacts.deduped_chunks))


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate CLI combinations that are unsafe for PostgreSQL replacement."""
    persist = not args.no_persist
    if persist and args.stage not in PERSISTABLE_STAGES:
        parser.error("PostgreSQL persistence requires --stage chunk_dedup or --stage all")
    if persist and args.limit is not None:
        parser.error(
            "--limit is only allowed with --no-persist to avoid replacing full data with a sample"
        )


def _build_chunk_token_counter(settings: Settings) -> Optional[ChunkTokenCounter]:
    """Build a token counter when an embedding model is configured."""
    embedding_model = settings.embedding.model
    if embedding_model is None:
        logger.warning(
            "EMBEDDING_MODEL is not configured; chunk token_count will be stored as NULL"
        )
        return None
    return ChunkTokenCounter.from_model_name(embedding_model)


async def _persist_after_pipeline(
    *,
    settings: Settings,
    artifacts: PipelineArtifacts,
    selected_crates: Sequence[str],
) -> IngestPersistenceResult:
    """Persist completed pipeline artifacts using one async database lifecycle."""
    db_engine = build_async_engine(settings.postgres)
    session_factory = build_session_factory(db_engine)
    try:
        if not await database_is_ready(session_factory):
            raise IngestDatabaseUnavailableError(
                "DATABASE_URL must point to a reachable PostgreSQL database"
            )
        token_counter = _build_chunk_token_counter(settings)
        return await persist_ingest_artifacts(
            artifacts=artifacts,
            session_factory=session_factory,
            replace_crates=selected_crates,
            token_counter=token_counter,
        )
    finally:
        await dispose_engine(db_engine)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ingest pipeline from command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    settings = get_settings()
    logging_settings = settings.logging
    if args.verbose:
        logging_settings = replace(logging_settings, level="DEBUG")
    configure_logging(logging_settings=logging_settings)

    raw_docs_dir = _resolve_raw_docs_dir(settings, parser)
    selected_crates = _selected_crates(args)
    persist = not args.no_persist
    logger.info(
        "Starting ingest pipeline stage=%s crates=%s persist_postgres=%s",
        args.stage,
        selected_crates,
        persist,
    )

    artifacts = run_pipeline_artifacts(
        raw_data_dir=raw_docs_dir,
        stage=args.stage,
        crates=selected_crates,
        limit=args.limit,
        max_chunk_chars=settings.ingest.max_chunk_chars,
        min_chunk_chars=settings.ingest.min_chunk_chars,
    )
    _log_stage_summary(args.stage, artifacts)

    if not persist:
        return 0

    try:
        persistence_result = asyncio.run(
            _persist_after_pipeline(
                settings=settings,
                artifacts=artifacts,
                selected_crates=selected_crates,
            )
        )
    except IngestDatabaseUnavailableError as exc:
        parser.error(str(exc))

    logger.info(
        "Persisted ingest status=%s docs=%s chunks=%s deleted_docs=%s deleted_chunks=%s",
        persistence_result.status,
        persistence_result.document_count,
        persistence_result.chunk_count,
        persistence_result.deleted_document_count,
        persistence_result.deleted_chunk_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
