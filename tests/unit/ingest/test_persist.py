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
    async def upsert_chunks(self, _session, chunks, _documents_by_source_path):
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


class FakeVectorStore:
    def __init__(self):
        self.payloads = []

    async def upsert_chunks(self, *, chunks):
        self.payloads = list(chunks)


class FailingVectorStore:
    async def upsert_chunks(self, *, chunks):
        _ = chunks
        raise RuntimeError("qdrant unavailable")


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


def test_persist_ingest_artifacts_uses_database_chunk_id_for_qdrant(monkeypatch):
    vector_store = FakeVectorStore()
    monkeypatch.setattr(
        "rust_assistant.ingest.persist.DocumentRepository",
        lambda: FakeDocumentRepository(),
    )
    monkeypatch.setattr(
        "rust_assistant.ingest.persist.ChunkRepository",
        lambda: FakeChunkRepository(),
    )

    result = asyncio.run(
        persist_ingest_artifacts(
            artifacts=PipelineArtifacts(
                deduped_docs=[_document()],
                deduped_chunks=[_chunk()],
            ),
            session_factory=FakeSessionFactory(),
            vector_store=vector_store,
        )
    )

    assert result.status == "completed"
    assert vector_store.payloads[0].chunk_id == 42


def test_persist_ingest_artifacts_reraises_qdrant_failures(monkeypatch):
    monkeypatch.setattr(
        "rust_assistant.ingest.persist.DocumentRepository",
        lambda: FakeDocumentRepository(),
    )
    monkeypatch.setattr(
        "rust_assistant.ingest.persist.ChunkRepository",
        lambda: FakeChunkRepository(),
    )

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        asyncio.run(
            persist_ingest_artifacts(
                artifacts=PipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[_chunk()],
                ),
                session_factory=FakeSessionFactory(),
                vector_store=FailingVectorStore(),
            )
        )
