"""CLI entry point for ingest pipeline execution."""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import replace
from pathlib import Path

from rust_assistant.clients.vectordb import StubVectorStoreClient
from rust_assistant.core.config import get_settings
from rust_assistant.core.db import build_async_engine, build_session_factory, dispose_engine
from rust_assistant.core.logging import configure_logging

from .persist import persist_ingest_artifacts
from .pipeline import run_pipeline, run_pipeline_artifacts

logger = logging.getLogger(__name__)


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
        "--raw-dir",
        type=Path,
        default="data/raw",
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--crate",
        type=str,
        action="append",
        choices=["std", "book", "cargo", "reference"],
        help="Crate(s) to include (can be specified multiple times, default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--parse_output",
        type=Path,
        default="data/processed/docs_parsed.jsonl",
        help="Output path for parsed documents (default: data/processed/docs_parsed.jsonl)",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default="data/processed/docs_cleaned.jsonl",
        help="Output path for cleaned documents (default: data/processed/docs_cleaned.jsonl)",
    )
    parser.add_argument(
        "--dedup-output",
        type=Path,
        default="data/processed/docs_deduped.jsonl",
        help="Output path for deduplicated documents (default: data/processed/docs_deduped.jsonl)",
    )
    parser.add_argument(
        "--chunk-output",
        type=Path,
        default="data/chunks/chunks.jsonl",
        help="Output path for chunks (default: data/chunks/chunks.jsonl)",
    )
    parser.add_argument(
        "--chunk-dedup-output",
        type=Path,
        default="data/chunks/chunks_deduped.jsonl",
        help="Output path for deduplicated chunks (default: data/chunks/chunks_deduped.jsonl)",
    )
    parser.add_argument(
        "--persist-postgres",
        action="store_true",
        help="Persist deduplicated documents and chunks to PostgreSQL after the pipeline run",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def _log_stage_summary(stage: str, result_count: int, args: argparse.Namespace) -> None:
    """Log a concise summary for the completed ingest stage."""
    if stage == "discover":
        logger.info("Discovered files: %s", result_count)
        return

    if stage == "parse":
        logger.info("Parsed documents: %s", result_count)
        logger.info("Saved parsed docs to: %s", args.parse_output)
        return

    if stage == "clean":
        logger.info("Cleaned documents: %s", result_count)
        logger.info("Saved parsed docs to: %s", args.parse_output)
        logger.info("Saved cleaned docs to: %s", args.clean_output)
        return

    if stage == "dedup":
        logger.info("Deduplicated documents: %s", result_count)
        logger.info("Saved parsed docs to: %s", args.parse_output)
        logger.info("Saved cleaned docs to: %s", args.clean_output)
        logger.info("Saved deduplicated docs to: %s", args.dedup_output)
        return

    if stage == "chunk":
        logger.info("Generated chunks: %s", result_count)
        logger.info("Saved parsed docs to: %s", args.parse_output)
        logger.info("Saved cleaned docs to: %s", args.clean_output)
        logger.info("Saved deduplicated docs to: %s", args.dedup_output)
        logger.info("Saved chunks to: %s", args.chunk_output)
        return

    logger.info("Deduplicated chunks: %s", result_count)
    logger.info("Saved parsed docs to: %s", args.parse_output)
    logger.info("Saved cleaned docs to: %s", args.clean_output)
    logger.info("Saved deduplicated docs to: %s", args.dedup_output)
    logger.info("Saved chunks to: %s", args.chunk_output)
    logger.info("Saved deduplicated chunks to: %s", args.chunk_dedup_output)


def main() -> int:
    """Run ingest pipeline from command line."""
    parser = build_parser()
    args = parser.parse_args()

    if args.persist_postgres and args.stage not in {"chunk_dedup", "all"}:
        parser.error("--persist-postgres currently requires --stage chunk_dedup or --stage all")

    settings = get_settings()
    logging_settings = settings.logging
    if args.verbose:
        logging_settings = replace(logging_settings, level="DEBUG")
    configure_logging(logging_settings=logging_settings)
    logger.info("Starting ingest pipeline stage=%s", args.stage)

    if args.persist_postgres:
        artifacts = run_pipeline_artifacts(
            stage=args.stage,
            raw_data_dir=args.raw_dir,
            crates=args.crate,
            limit=args.limit,
            parsing_output=args.parse_output,
            clean_output=args.clean_output,
            dedup_output=args.dedup_output,
            chunk_output=args.chunk_output,
            chunk_dedup_output=args.chunk_dedup_output,
        )
        _log_stage_summary(args.stage, len(artifacts.deduped_chunks), args)

        db_engine = build_async_engine(settings.postgres)
        session_factory = build_session_factory(db_engine)
        try:
            persistence_result = asyncio.run(
                persist_ingest_artifacts(
                    artifacts=artifacts,
                    session_factory=session_factory,
                    vector_store=StubVectorStoreClient(),
                )
            )
        finally:
            asyncio.run(dispose_engine(db_engine))

        logger.info(
            "Persisted ingest status=%s docs=%s chunks=%s synced=%s failed=%s",
            persistence_result.status,
            persistence_result.document_count,
            persistence_result.chunk_count,
            persistence_result.synced_chunk_count,
            persistence_result.failed_chunk_count,
        )
        return 0

    result = run_pipeline(
        stage=args.stage,
        raw_data_dir=args.raw_dir,
        crates=args.crate,
        limit=args.limit,
        parsing_output=args.parse_output,
        clean_output=args.clean_output,
        dedup_output=args.dedup_output,
        chunk_output=args.chunk_output,
        chunk_dedup_output=args.chunk_dedup_output,
    )

    _log_stage_summary(args.stage, len(result), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
