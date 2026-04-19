"""
Parse discovered HTML files into `Document` records.

This module implements ingest stage 1.3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from rust_assistant.ingest.entities import Document
from rust_assistant.ingest.parsing.page_parser import PageParser

logger = logging.getLogger(__name__)


def parse(
    html_files: list[Path],
    raw_data_dir: Union[Path, str],
) -> list[Document]:
    """
    Parse discovered HTML files into `Document` instances.

    Args:
        html_files: List of HTML file paths from discovery stage.
        raw_data_dir: Root directory used to build relative source paths.

    Returns:
        List of successfully parsed documents.
    """
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

    return docs
