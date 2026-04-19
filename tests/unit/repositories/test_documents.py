import asyncio
from typing import Optional

import pytest

from rust_assistant.ingest.entities import (
    BlockType,
    Document,
    DocumentMetadata,
    StructuredBlock,
)
from rust_assistant.repositories.documents import DocumentRepository
from rust_assistant.schemas.enums import Crate, ItemType

pytestmark = pytest.mark.unit


class FakeSession:
    def __init__(self, existing=None):
        self.existing = existing or []
        self.added = []
        self.flushed = False

    async def scalars(self, _statement):
        return self.existing

    def add(self, record):
        self.added.append(record)

    async def flush(self):
        self.flushed = True
        for index, record in enumerate(self.added, start=1):
            record.id = index


class FakeDeleteSession:
    def __init__(self):
        self.scalar_results = [2, 5]
        self.executed = []
        self.flushed = False

    async def scalar(self, _statement):
        return self.scalar_results.pop(0)

    async def execute(self, statement):
        self.executed.append(statement)

    async def flush(self):
        self.flushed = True


def _document(url: Optional[str] = "https://doc.rust-lang.org/std/keyword.async.html") -> Document:
    return Document(
        doc_id="transient-doc-id",
        title="std::keyword::async",
        source_path="std/keyword.async.html",
        text="Keyword async\n\nReturns a Future.",
        structured_blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Keyword async",
                html_tag="h1",
                anchor="keyword.async",
                section_path=["std::keyword::async"],
            )
        ],
        metadata=DocumentMetadata(
            crate=Crate.STD,
            item_path="std::keyword::async",
            item_type=ItemType.UNKNOWN,
            rust_version="1.91.1",
            url=url,
        ),
    )


def test_upsert_documents_maps_ingest_document_to_new_schema():
    session = FakeSession()
    repository = DocumentRepository()

    documents_by_source_path = asyncio.run(repository.upsert_documents(session, [_document()]))

    record = documents_by_source_path["std/keyword.async.html"]
    assert session.flushed is True
    assert record.source_path == "std/keyword.async.html"
    assert record.text_content == "Keyword async\n\nReturns a Future."
    assert record.parsed_content[0]["block_type"] == "heading"
    assert record.parsed_content[0]["anchor"] == "keyword.async"
    assert record.url == "https://doc.rust-lang.org/std/keyword.async.html"
    assert record.crate == "std"
    assert record.item_path == "std::keyword::async"
    assert record.item_type == "unknown"
    assert record.rust_version == "1.91.1"


def test_upsert_documents_rejects_missing_required_url():
    session = FakeSession()
    repository = DocumentRepository()

    with pytest.raises(ValueError, match="Document URL is required"):
        asyncio.run(repository.upsert_documents(session, [_document(url=None)]))


def test_delete_by_crates_returns_deleted_document_and_chunk_counts():
    session = FakeDeleteSession()
    repository = DocumentRepository()

    counts = asyncio.run(repository.delete_by_crates(session, ["std"]))

    assert counts == (2, 5)
    assert len(session.executed) == 1
    assert session.flushed is True
