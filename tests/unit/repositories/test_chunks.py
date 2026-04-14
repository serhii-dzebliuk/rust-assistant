import asyncio

import pytest

from rust_assistant.ingest.entities import Chunk, ChunkMetadata
from rust_assistant.models import DocumentRecord
from rust_assistant.repositories.chunks import ChunkRepository
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
        for index, record in enumerate(self.added, start=42):
            record.id = index


def _document_record() -> DocumentRecord:
    record = DocumentRecord(
        source_path="std/keyword.async.html",
        crate="std",
        title="std::keyword::async",
        text_content="Keyword async",
        parsed_content=[],
        url="https://doc.rust-lang.org/std/keyword.async.html",
    )
    record.id = 7
    return record


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="transient-chunk-id",
        doc_id="transient-doc-id",
        text="Returns a Future instead of blocking the current thread.",
        metadata=ChunkMetadata(
            crate=Crate.STD,
            item_path="std::keyword::async",
            item_type=ItemType.UNKNOWN,
            rust_version="1.91.1",
            url="https://doc.rust-lang.org/std/keyword.async.html",
            section="Keyword async",
            section_path=["std::keyword::async", "Keyword async"],
            anchor="keyword.async",
            chunk_index=2,
            start_char=128,
            end_char=192,
            doc_title="std::keyword::async",
            doc_source_path="std/keyword.async.html",
        ),
        text_hash=None,
    )


def test_upsert_chunks_maps_ingest_chunk_to_new_schema_and_returns_ordered_records():
    session = FakeSession()
    repository = ChunkRepository()

    records = asyncio.run(
        repository.upsert_chunks(
            session,
            [_chunk()],
            {"std/keyword.async.html": _document_record()},
        )
    )

    assert session.flushed is True
    assert len(records) == 1
    record = records[0]
    assert record.id == 42
    assert record.document_id == 7
    assert record.text == "Returns a Future instead of blocking the current thread."
    assert record.hash == Chunk.compute_text_hash(record.text)
    assert record.token_count is None
    assert record.section_title == "Keyword async"
    assert record.section_anchor == "keyword.async"
    assert record.section_path == ["std::keyword::async", "Keyword async"]
    assert record.chunk_index == 2
    assert record.start_offset == 128
    assert record.end_offset == 192
