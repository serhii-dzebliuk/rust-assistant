from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RustRAG ingest pipeline")
    parser.add_argument(
        "--stage",
        choices=["discover", "parse", "clean", "dedup", "all"],
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
        "--docs-output",
        type=Path,
        default="data/processed/docs.jsonl",
        help="Output path for parsed documents (default: data/processed/docs.jsonl)",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default="data/processed/docs_clean.jsonl",
        help="Output path for cleaned documents (default: data/processed/docs_clean.jsonl)",
    )
    parser.add_argument(
        "--dedup-output",
        type=Path,
        default="data/processed/docs_dedup.jsonl",
        help="Output path for deduplicated documents (default: data/processed/docs_dedup.jsonl)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = run_pipeline(
        stage=args.stage,
        raw_data_dir=args.raw_dir,
        crates=args.crate,
        limit=args.limit,
        docs_output=args.docs_output,
        clean_output=args.clean_output,
        dedup_output=args.dedup_output,
    )

    if args.stage == "discover":
        print(f"Discovered files: {len(result)}")
    elif args.stage == "parse":
        print(f"Parsed documents: {len(result)}")
        print(f"Saved parsed docs to: {args.docs_output}")
    elif args.stage == "clean":
        print(f"Cleaned documents: {len(result)}")
        print(f"Saved parsed docs to: {args.docs_output}")
        print(f"Saved cleaned docs to: {args.clean_output}")
    else:
        print(f"Deduplicated documents: {len(result)}")
        print(f"Saved parsed docs to: {args.docs_output}")
        print(f"Saved cleaned docs to: {args.clean_output}")
        print(f"Saved deduplicated docs to: {args.dedup_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
