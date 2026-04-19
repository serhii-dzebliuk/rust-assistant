import asyncio

import pytest

from rust_assistant.ingest.entities import Chunk, ChunkMetadata, Document, DocumentMetadata
from rust_assistant.ingest.pipeline import PipelineArtifacts
from rust_assistant.ingest.persist import persist_ingest_artifacts
from rust_assistant.models import ChunkRecord, DocumentRecord
from rust_assistant.schemas.enums import Crate, ItemType

pytestmark = pytest.mark.unit


class FakeSession:
    def begin(self):
        return FakeTransactionContext()


class FakeTransactionContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSessionContext:
    async def __aenter__(self):
        return FakeSession()

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSessionFactory:
    def __call__(self):
        return FakeSessionContext()


class FakeDocumentRepository:
    def __init__(self):
        self.deleted_crates = None

    async def delete_by_crates(self, _session, crate_values):
        self.deleted_crates = list(crate_values)
        return (3, 9)

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
    document_repository = FakeDocumentRepository()
    monkeypatch.setattr(
        "rust_assistant.ingest.persist.DocumentRepository",
        lambda: document_repository,
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
            replace_crates=["std"],
        )
    )

    assert result.status == "completed"
    assert result.document_count == 1
    assert result.chunk_count == 1
    assert result.deleted_document_count == 3
    assert result.deleted_chunk_count == 9
    assert document_repository.deleted_crates == ["std"]
    assert chunk_repository.chunks == [_chunk()]
    assert "std/keyword.async.html" in chunk_repository.documents_by_source_path


def test_persist_ingest_artifacts_refuses_empty_replace_payload():
    with pytest.raises(ValueError, match="zero documents"):
        asyncio.run(
            persist_ingest_artifacts(
                artifacts=PipelineArtifacts(),
                session_factory=FakeSessionFactory(),
                replace_crates=["std"],
            )
        )


def test_persist_ingest_artifacts_refuses_documents_without_chunks():
    chunk = _chunk().model_copy(
        update={
            "metadata": _chunk().metadata.model_copy(update={"doc_source_path": "std/other.html"})
        }
    )

    with pytest.raises(ValueError, match="documents without chunks"):
        asyncio.run(
            persist_ingest_artifacts(
                artifacts=PipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[chunk],
                ),
                session_factory=FakeSessionFactory(),
                replace_crates=["std"],
            )
        )


def test_persist_ingest_artifacts_refuses_chunks_without_matching_documents():
    chunk = _chunk().model_copy(
        update={
            "metadata": _chunk().metadata.model_copy(update={"doc_source_path": "std/missing.html"})
        }
    )

    with pytest.raises(ValueError, match="chunks without matching documents"):
        asyncio.run(
            persist_ingest_artifacts(
                artifacts=PipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[_chunk(), chunk],
                ),
                session_factory=FakeSessionFactory(),
                replace_crates=["std"],
            )
        )
