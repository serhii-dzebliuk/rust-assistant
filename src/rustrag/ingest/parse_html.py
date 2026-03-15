"""
Parse HTML documentation files into Document objects.
Stage 1.3 of the ingest pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rustrag.ingest.parsing.page_parser import PageParser
from ..models import Document

logger = logging.getLogger(__name__)

def parse_documents(
    html_files: list[Path],
    raw_data_dir: Path | str = "data/raw",
    output_file: Optional[Path | str] = None,
) -> list[Document]:
    parser = PageParser(raw_data_dir)
    docs: list[Document] = []
    failed = 0

    logger.info("Parsing %s HTML files...", len(html_files))
    for idx, file_path in enumerate(html_files, start=1):
        if idx % 100 == 0:
            logger.info("Parsed %s/%s files (%s failed)", idx, len(html_files), failed)
        doc = parser.parse_file(file_path)
        if doc is None:
            failed += 1
            continue
        docs.append(doc)

    logger.info("Parsed %s documents (%s failed)", len(docs), failed)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc.to_jsonl() + "\n")
        logger.info("Saved parsed documents to %s", output_path)

    return docs


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Parse Rust HTML documentation into Documents")
    parser.add_argument("--raw-dir", type=Path, default="data/raw")
    parser.add_argument("--output", "-o", type=Path, default="data/processed/docs.jsonl")
    parser.add_argument("--crate", type=str, action="append", choices=["std", "book", "cargo", "reference"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from .discover import discover_documents

    html_files = discover_documents(raw_data_dir=args.raw_dir, crates=args.crate, limit=args.limit)
    if not html_files:
        logger.error("No HTML files found")
        return 1

    docs = parse_documents(html_files, raw_data_dir=args.raw_dir, output_file=args.output)
    print(f"Total files: {len(html_files)}")
    print(f"Parsed successfully: {len(docs)}")
    print(f"Failed: {len(html_files) - len(docs)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
