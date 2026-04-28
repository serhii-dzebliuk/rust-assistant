from pathlib import Path

import pytest

from rust_assistant.application.dto.document_parse import (
    DocumentParseFailure,
    DocumentParseFailureReason,
    DocumentParseResult,
)
from rust_assistant.application.use_cases.ingest.parse_documents import (
    ParseDocumentsCommand,
    ParseDocumentsUseCase,
)
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate

pytestmark = pytest.mark.unit


class FakeDocumentParser:
    def __init__(self, results):
        self.results = list(results)
        self.files = []

    def parse_file(self, file_path):
        self.files.append(file_path)
        return self.results.pop(0)


def test_parse_documents_returns_docs_and_structured_failures():
    document = Document(
        source_path="std/index.html",
        title="std",
        text="Standard library",
        crate=Crate.STD,
        url="https://doc.rust-lang.org/std/index.html",
    )
    failure = DocumentParseFailure(
        file_path=Path("missing.html"),
        reason=DocumentParseFailureReason.MISSING_MAIN_CONTENT,
        message="No main content found",
    )
    parser = FakeDocumentParser(
        [
            DocumentParseResult.success(document),
            DocumentParseResult(failure=failure),
        ]
    )

    result = ParseDocumentsUseCase(parser).execute(
        ParseDocumentsCommand(html_files=[Path("std/index.html"), Path("missing.html")])
    )

    assert parser.files == [Path("std/index.html"), Path("missing.html")]
    assert result.documents == [document]
    assert result.failures == [failure]
