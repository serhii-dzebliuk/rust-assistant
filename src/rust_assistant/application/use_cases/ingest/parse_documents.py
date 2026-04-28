"""Use case for parsing discovered HTML files into canonical documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rust_assistant.application.dto.document_parse import (
    DocumentParseFailure,
)
from rust_assistant.application.ports.ingest.document_parser import DocumentParser
from rust_assistant.domain.entities.documents import Document

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ParseDocumentsCommand:
    """Input for parsing discovered raw HTML files."""

    html_files: list[Path]


@dataclass(slots=True, frozen=True)
class ParseDocumentsResult:
    """Batch parse result produced by the parse-documents use case."""

    documents: list[Document]
    failures: list[DocumentParseFailure]


class ParseDocumentsUseCase:
    """Parse discovered raw HTML files into canonical documents."""

    def __init__(self, parser: DocumentParser):
        self._parser = parser

    def execute(self, command: ParseDocumentsCommand) -> ParseDocumentsResult:
        """Parse discovered HTML files into canonical document objects."""
        docs: list[Document] = []
        failures: list[DocumentParseFailure] = []

        html_files = command.html_files
        logger.info("Parsing %s HTML files...", len(html_files))
        for idx, file_path in enumerate(html_files, start=1):
            if idx % 100 == 0:
                logger.info(
                    "Parsed %s/%s files (%s failed)",
                    idx,
                    len(html_files),
                    len(failures),
                )
            result = self._parser.parse_file(file_path)
            if result.failure is not None:
                failures.append(result.failure)
                continue
            if result.document is not None:
                docs.append(result.document)

        logger.info("Parsed %s documents (%s failed)", len(docs), len(failures))
        return ParseDocumentsResult(documents=docs, failures=failures)
