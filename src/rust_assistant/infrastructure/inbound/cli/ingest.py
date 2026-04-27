"""CLI transport adapter for the ingest runtime."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

SUPPORTED_CRATES = ("std", "book", "cargo", "reference")


def _add_ingest_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach ingest-specific CLI arguments to a parser."""
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


def build_parser() -> argparse.ArgumentParser:
    """Build a standalone parser for the ingest command."""
    parser = argparse.ArgumentParser(description="Run the rust-assistant ingest pipeline")
    return _add_ingest_arguments(parser)


def register_ingest_subcommand(subparsers: Any) -> None:
    """Register the ingest subcommand on the root CLI."""
    parser = subparsers.add_parser("ingest", help="Run the ingest pipeline")
    _add_ingest_arguments(parser)


def ingest_kwargs_from_args(args: argparse.Namespace) -> Mapping[str, object]:
    """Map parsed CLI args into bootstrap ingest runtime keyword arguments."""
    return {
        "stage": args.stage,
        "crates": args.crate,
        "limit": args.limit,
        "persist": not args.no_persist,
        "verbose": args.verbose,
    }
