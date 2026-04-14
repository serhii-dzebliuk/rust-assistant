import asyncio

import pytest

from rust_assistant.ingest.entities import Chunk, ChunkMetadata, Document, DocumentMetadata
from rust_assistant.ingest.pipeline import PipelineArtifacts
from rust_assistant.ingest.persist import persist_ingest_artifacts
from rust_assistant.models import ChunkRecord, DocumentRecord
from rust_assistant.schemas.enums import Crate, ItemType

pytestmark = pytest.mark.unit


class FakeSession:
    async def commit(self):
        return None


class FakeSessionContext:
    async def __aenter__(self):
        return FakeSession()

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSessionFactory:
    def __call__(self):
        return FakeSessionContext()


class FakeDocumentRepository:
    async def upsert_documents(self, _session, _documents):
        record = DocumentRecord(
            source_path="std/keyword.async.html",
            crate="std",
            title="std::keyword::async",
            text_content="Keyword async",
            parsed_content=[],
            url="https://doc.rust-lang.org/std/keyword.async.html",
        )
        record.id = 7
        return {"std/keyword.async.html": record}


class FakeChunkRepository:
    def __init__(self):
        self.chunks = None
        self.documents_by_source_path = None

    async def upsert_chunks(self, _session, chunks, _documents_by_source_path):
        self.chunks = list(chunks)
        self.documents_by_source_path = dict(_documents_by_source_path)
        records = []
        for index, chunk in enumerate(chunks, start=42):
            record = ChunkRecord(
                document_id=7,
                text=chunk.text,
                hash=chunk.text_hash or Chunk.compute_text_hash(chunk.text),
                chunk_index=chunk.metadata.chunk_index,
            )
            record.id = index
            records.append(record)
        return records


def _document() -> Document:
    return Document(
        doc_id="transient-doc-id",
        title="std::keyword::async",
        source_path="std/keyword.async.html",
        text="Keyword async",
        metadata=DocumentMetadata(
            crate=Crate.STD,
            item_path="std::keyword::async",
            item_type=ItemType.UNKNOWN,
            rust_version="1.91.1",
            url="https://doc.rust-lang.org/std/keyword.async.html",
        ),
    )


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="transient-chunk-id",
        doc_id="transient-doc-id",
        text="Returns a Future.",
        metadata=ChunkMetadata(
            crate=Crate.STD,
            item_path="std::keyword::async",
            item_type=ItemType.UNKNOWN,
            rust_version="1.91.1",
            url="https://doc.rust-lang.org/std/keyword.async.html",
            section="Keyword async",
            section_path=["std::keyword::async"],
            anchor="keyword.async",
            chunk_index=0,
            start_char=0,
            end_char=17,
            doc_title="std::keyword::async",
            doc_source_path="std/keyword.async.html",
        ),
    )


def test_persist_ingest_artifacts_persists_documents_and_chunks_to_postgres(monkeypatch):
    chunk_repository = FakeChunkRepository()
    monkeypatch.setattr(
        "rust_assistant.ingest.persist.DocumentRepository",
        lambda: FakeDocumentRepository(),
    )
    monkeypatch.setattr(
        "rust_assistant.ingest.persist.ChunkRepository",
        lambda: chunk_repository,
    )

    result = asyncio.run(
        persist_ingest_artifacts(
            artifacts=PipelineArtifacts(
                deduped_docs=[_document()],
                deduped_chunks=[_chunk()],
            ),
            session_factory=FakeSessionFactory(),
        )
    )

    assert result.status == "completed"
    assert result.document_count == 1
    assert result.chunk_count == 1
    assert chunk_repository.chunks == [_chunk()]
    assert "std/keyword.async.html" in chunk_repository.documents_by_source_path
