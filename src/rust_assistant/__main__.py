"""Public package CLI entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Optional

from rust_assistant.bootstrap.ingest import (
    IngestConfigurationError,
    IngestDatabaseUnavailableError,
    IngestTokenizerUnavailableError,
    run_ingest,
)
from rust_assistant.infrastructure.entrypoints.cli.ingest import (
    ingest_kwargs_from_args,
    register_ingest_subcommand,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser."""
    parser = argparse.ArgumentParser(prog="rust-assistant", description="Rust Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_ingest_subcommand(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the package CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        try:
            return run_ingest(**ingest_kwargs_from_args(args))
        except (
            IngestConfigurationError,
            IngestDatabaseUnavailableError,
            IngestTokenizerUnavailableError,
            ValueError,
        ) as exc:
            parser.error(str(exc))

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
