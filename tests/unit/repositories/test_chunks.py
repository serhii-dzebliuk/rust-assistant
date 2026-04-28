import asyncio
from dataclasses import replace

import pytest

from rust_assistant.application.dto.chunk_context import ChunkContext
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.domain.value_objects.identifiers import build_chunk_id, build_document_id
from rust_assistant.infrastructure.adapters.sqlalchemy.mappers import map_chunk_to_domain
from rust_assistant.infrastructure.adapters.sqlalchemy.models import ChunkRecord, DocumentRecord
from rust_assistant.infrastructure.adapters.sqlalchemy.repositories.chunk_repository import (
    SqlAlchemyChunkRepository,
)

pytestmark = pytest.mark.unit


class FakeSession:
    def __init__(self, *, execute_result=None, scalar_result=None, scalars_result=None):
        self.execute_result = execute_result or []
        self.scalar_result = scalar_result
        self.scalars_result = scalars_result or []
        self.added = []
        self.flushed = False

    def add_all(self, records):
        self.added.extend(records)

    async def execute(self, _statement):
        return self.execute_result

    async def scalar(self, _statement):
        return self.scalar_result

    async def scalars(self, _statement):
        return self.scalars_result

    async def flush(self):
        self.flushed = True


def _chunk() -> Chunk:
    return Chunk(
        source_path="std/keyword.async.html",
        chunk_index=2,
        text="Returns a Future instead of blocking the current thread.",
        crate=Crate.STD,
        start_offset=128,
        end_offset=192,
        item_path="std::keyword::async",
        item_type=ItemType.UNKNOWN,
        rust_version="1.91.1",
        url="https://doc.rust-lang.org/std/keyword.async.html",
        section_path=["std::keyword::async", "Keyword async"],
        section_anchor="keyword.async",
        token_count=9,
    )


def _document_record_for(chunk: Chunk, pk: int = 7) -> DocumentRecord:
    return DocumentRecord(
        pk=pk,
        id=chunk.document_id,
        source_path=chunk.source_path,
        crate=chunk.crate.value,
        title="std::keyword::async",
        text_content="Keyword async",
        parsed_content=[],
        url=chunk.url or "https://doc.rust-lang.org/std/keyword.async.html",
        item_path=chunk.item_path,
        item_type=chunk.item_type.value if chunk.item_type else None,
        rust_version=chunk.rust_version,
    )


def test_add_many_maps_ingest_chunk_to_new_schema():
    chunk = _chunk()
    session = FakeSession(execute_result=[(chunk.document_id, 7)])
    repository = SqlAlchemyChunkRepository(session)

    asyncio.run(repository.add_many([chunk]))

    assert session.flushed is True
    record = session.added[0]
    assert record.id == chunk.id
    assert record.document_pk == 7
    assert record.text == "Returns a Future instead of blocking the current thread."
    assert record.hash == Chunk.compute_text_hash(record.text)
    assert record.token_count == 9
    assert record.section_title == "Keyword async"
    assert record.section_anchor == "keyword.async"
    assert record.section_path == ["std::keyword::async", "Keyword async"]
    assert record.chunk_index == 2
    assert record.start_offset == 128
    assert record.end_offset == 192


def test_add_many_derives_section_title_from_section_path():
    chunk = replace(_chunk(), section_path=("std::keyword::async", "Examples"))
    session = FakeSession(execute_result=[(chunk.document_id, 7)])
    repository = SqlAlchemyChunkRepository(session)

    asyncio.run(repository.add_many([chunk]))

    assert session.added[0].section_title == "Examples"


def test_add_is_thin_wrapper_over_add_many():
    chunk = _chunk()
    session = FakeSession(execute_result=[(chunk.document_id, 7)])
    repository = SqlAlchemyChunkRepository(session)

    asyncio.run(repository.add(chunk))

    assert len(session.added) == 1
    assert session.added[0].id == chunk.id


def test_get_returns_chunk_by_business_id():
    chunk = _chunk()
    document = _document_record_for(chunk)
    record = ChunkRecord(
        pk=42,
        id=chunk.id,
        document_pk=document.pk,
        text=chunk.text,
        hash=chunk.text_hash,
        token_count=chunk.token_count,
        section_title=chunk.section_title,
        section_anchor=chunk.section_anchor,
        section_path=list(chunk.section_path),
        chunk_index=chunk.chunk_index,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
    )
    record.document = document
    session = FakeSession(scalar_result=record)
    repository = SqlAlchemyChunkRepository(session)

    loaded = asyncio.run(repository.get(chunk.id))

    assert loaded == chunk


def test_map_chunk_to_domain_raises_on_chunk_identity_mismatch():
    chunk = _chunk()
    document = _document_record_for(chunk)
    record = ChunkRecord(
        pk=42,
        id=build_chunk_id(build_document_id("std/other.html"), chunk.chunk_index),
        document_pk=document.pk,
        text=chunk.text,
        hash=chunk.text_hash,
        token_count=chunk.token_count,
        section_title=chunk.section_title,
        section_anchor=chunk.section_anchor,
        section_path=list(chunk.section_path),
        chunk_index=chunk.chunk_index,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
    )
    record.document = document

    with pytest.raises(ValueError, match="Chunk identity mismatch"):
        map_chunk_to_domain(record)


def test_map_chunk_to_domain_raises_on_parent_identity_mismatch():
    chunk = _chunk()
    document = _document_record_for(chunk)
    document.id = build_document_id("std/other.html")
    record = ChunkRecord(
        pk=42,
        id=chunk.id,
        document_pk=document.pk,
        text=chunk.text,
        hash=chunk.text_hash,
        token_count=chunk.token_count,
        section_title=chunk.section_title,
        section_anchor=chunk.section_anchor,
        section_path=list(chunk.section_path),
        chunk_index=chunk.chunk_index,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
    )
    record.document = document

    with pytest.raises(ValueError, match="Chunk parent document identity mismatch"):
        map_chunk_to_domain(record)


def test_get_contexts_preserves_requested_order():
    first = _chunk()
    second = replace(_chunk(), chunk_index=3, section_path=("std::keyword::async", "Examples"))
    first_document = _document_record_for(first, pk=7)
    second_document = _document_record_for(second, pk=8)
    first_record = ChunkRecord(
        pk=101,
        id=first.id,
        document_pk=first_document.pk,
        text=first.text,
        hash=first.text_hash,
        token_count=first.token_count,
        section_title=first.section_title,
        section_anchor=first.section_anchor,
        section_path=list(first.section_path),
        chunk_index=first.chunk_index,
        start_offset=first.start_offset,
        end_offset=first.end_offset,
    )
    first_record.document = first_document
    second_record = ChunkRecord(
        pk=102,
        id=second.id,
        document_pk=second_document.pk,
        text=second.text,
        hash=second.text_hash,
        token_count=second.token_count,
        section_title=second.section_title,
        section_anchor=second.section_anchor,
        section_path=list(second.section_path),
        chunk_index=second.chunk_index,
        start_offset=second.start_offset,
        end_offset=second.end_offset,
    )
    second_record.document = second_document
    session = FakeSession(scalars_result=[first_record, second_record])
    repository = SqlAlchemyChunkRepository(session)

    contexts = asyncio.run(repository.get_contexts([second.id, first.id]))

    assert contexts == [
        ChunkContext(
            chunk_id=second.id,
            document_id=second.document_id,
            text=second.text,
            title="std::keyword::async",
            source_path=second.source_path,
            url=second.url or "",
            section_title="Examples",
            section_path=("std::keyword::async", "Examples"),
            section_anchor="keyword.async",
            item_path=second.item_path,
            crate=second.crate,
            item_type=second.item_type,
            rust_version=second.rust_version,
            chunk_index=second.chunk_index,
        ),
        ChunkContext(
            chunk_id=first.id,
            document_id=first.document_id,
            text=first.text,
            title="std::keyword::async",
            source_path=first.source_path,
            url=first.url or "",
            section_title="Keyword async",
            section_path=("std::keyword::async", "Keyword async"),
            section_anchor="keyword.async",
            item_path=first.item_path,
            crate=first.crate,
            item_type=first.item_type,
            rust_version=first.rust_version,
            chunk_index=first.chunk_index,
        ),
    ]
