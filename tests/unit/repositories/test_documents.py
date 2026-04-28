import asyncio
from typing import Optional

import pytest

from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.domain.value_objects.identifiers import build_document_id
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.mappers import map_document_to_domain
from rust_assistant.domain.value_objects.structured_blocks import (
    BlockType,
    StructuredBlock,
)
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.models import DocumentRecord
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.repositories.document_repository import (
    SqlAlchemyDocumentRepository,
)

pytestmark = pytest.mark.unit


class FakeSession:
    def __init__(self, scalar_result=None):
        self.scalar_result = scalar_result
        self.added = []
        self.executed = []
        self.flushed = False

    def add_all(self, records):
        self.added.extend(records)

    async def scalar(self, _statement):
        return self.scalar_result

    async def execute(self, statement):
        self.executed.append(statement)

    async def flush(self):
        self.flushed = True


def _document(url: Optional[str] = "https://doc.rust-lang.org/std/keyword.async.html") -> Document:
    if url is None:
        raise ValueError("Document url cannot be empty")
    return Document(
        source_path="std/keyword.async.html",
        title="std::keyword::async",
        text="Keyword async\n\nReturns a Future.",
        crate=Crate.STD,
        url=url,
        item_path="std::keyword::async",
        item_type=ItemType.UNKNOWN,
        rust_version="1.91.1",
        structured_blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Keyword async",
                html_tag="h1",
                anchor="keyword.async",
                section_path=["std::keyword::async"],
            )
        ],
    )


def test_add_many_maps_ingest_document_to_new_schema():
    session = FakeSession()
    repository = SqlAlchemyDocumentRepository(session)

    asyncio.run(repository.add_many([_document()]))

    assert session.flushed is True
    record = session.added[0]
    assert record.id == build_document_id("std/keyword.async.html")
    assert record.source_path == "std/keyword.async.html"
    assert record.text_content == "Keyword async\n\nReturns a Future."
    assert record.parsed_content[0]["block_type"] == "heading"
    assert record.parsed_content[0]["anchor"] == "keyword.async"
    assert record.url == "https://doc.rust-lang.org/std/keyword.async.html"
    assert record.crate == "std"
    assert record.item_path == "std::keyword::async"
    assert record.item_type == "unknown"
    assert record.rust_version == "1.91.1"


def test_document_requires_url():
    with pytest.raises(ValueError, match="Document url cannot be empty"):
        _document(url=None)


def test_add_is_thin_wrapper_over_add_many():
    session = FakeSession()
    repository = SqlAlchemyDocumentRepository(session)

    asyncio.run(repository.add(_document()))

    assert len(session.added) == 1
    assert session.added[0].id == build_document_id("std/keyword.async.html")


def test_get_returns_document_by_business_id():
    document = _document()
    record = DocumentRecord(
        id=document.id,
        source_path=document.source_path,
        crate=document.crate.value,
        title=document.title,
        text_content=document.text,
        parsed_content=[
            {
                "block_type": "heading",
                "text": "Keyword async",
                "html_tag": "h1",
                "heading_level": None,
                "list_depth": None,
                "code_language": None,
                "anchor": "keyword.async",
                "section_path": ["std::keyword::async"],
            }
        ],
        url=document.url,
        item_path=document.item_path,
        item_type=document.item_type.value if document.item_type else None,
        rust_version=document.rust_version,
    )
    session = FakeSession(scalar_result=record)
    repository = SqlAlchemyDocumentRepository(session)

    loaded = asyncio.run(repository.get(document.id))

    assert loaded == document


def test_map_document_to_domain_raises_on_identity_mismatch():
    document = _document()
    record = DocumentRecord(
        id=build_document_id("std/other.html"),
        source_path=document.source_path,
        crate=document.crate.value,
        title=document.title,
        text_content=document.text,
        parsed_content=[],
        url=document.url,
        item_path=document.item_path,
        item_type=document.item_type.value if document.item_type else None,
        rust_version=document.rust_version,
    )

    with pytest.raises(ValueError, match="Document identity mismatch"):
        map_document_to_domain(record)


def test_delete_all_executes_delete_and_flushes():
    session = FakeSession()
    repository = SqlAlchemyDocumentRepository(session)

    asyncio.run(repository.delete_all())

    assert len(session.executed) == 1
    assert session.flushed is True
