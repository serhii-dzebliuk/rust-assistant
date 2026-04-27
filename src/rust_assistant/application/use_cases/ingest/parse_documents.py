"""Use case for parsing discovered HTML files into canonical documents."""

from __future__ import annotations

import logging
from pathlib import Path

from rust_assistant.application.ports.document_parser import DocumentParserPort
from rust_assistant.domain.entities.documents import Document

logger = logging.getLogger(__name__)


class ParseDocuments:
    """Parse discovered raw HTML files into canonical documents."""

    def __init__(self, parser: DocumentParserPort):
        self._parser = parser

    def execute(self, *, html_files: list[Path]) -> list[Document]:
        """Parse discovered HTML files into canonical document objects."""
        docs: list[Document] = []
        failed = 0

        logger.info("Parsing %s HTML files...", len(html_files))
        for idx, file_path in enumerate(html_files, start=1):
            if idx % 100 == 0:
                logger.info("Parsed %s/%s files (%s failed)", idx, len(html_files), failed)
            doc = self._parser.parse_file(file_path)
            if doc is None:
                failed += 1
                continue
            docs.append(doc)

        logger.info("Parsed %s documents (%s failed)", len(docs), failed)
        return docs
