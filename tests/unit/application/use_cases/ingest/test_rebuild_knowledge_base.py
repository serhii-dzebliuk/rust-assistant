import asyncio
from dataclasses import replace

import pytest

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.rebuild_knowledge_base import (
    RebuildKnowledgeBase,
)
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate, ItemType

pytestmark = pytest.mark.unit


class FakeDocumentRepository:
    def __init__(self):
        self.deleted_all = False
        self.documents = None

    async def delete_all(self):
        self.deleted_all = True

    async def add_many(self, documents):
        self.documents = list(documents)


class FakeChunkRepository:
    def __init__(self):
        self.chunks = None

    async def add_many(self, chunks):
        self.chunks = list(chunks)


class FakeUnitOfWork:
    def __init__(self, document_repository, chunk_repository):
        self.documents = document_repository
        self.chunks = chunk_repository
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def commit(self):
        self.committed = True

    async def rollback(self):
        return None


class FakeTokenCounter:
    def __init__(self, token_count):
        self.token_count = token_count
        self.chunks = None

    def with_token_counts(self, chunks):
        self.chunks = list(chunks)
        return [replace(chunk, token_count=self.token_count) for chunk in chunks]


def _document() -> Document:
    return Document(
        source_path="std/keyword.async.html",
        title="std::keyword::async",
        text="Keyword async",
        crate=Crate.STD,
        url="https://doc.rust-lang.org/std/keyword.async.html",
        item_path="std::keyword::async",
        item_type=ItemType.UNKNOWN,
        rust_version="1.91.1",
    )


def _chunk() -> Chunk:
    return Chunk(
        source_path="std/keyword.async.html",
        chunk_index=0,
        text="Returns a Future.",
        crate=Crate.STD,
        start_offset=0,
        end_offset=17,
        item_path="std::keyword::async",
        item_type=ItemType.UNKNOWN,
        rust_version="1.91.1",
        url="https://doc.rust-lang.org/std/keyword.async.html",
        section_path=["std::keyword::async"],
        section_anchor="keyword.async",
    )


def test_persist_ingest_artifacts_persists_documents_and_chunks_to_postgres():
    chunk_repository = FakeChunkRepository()
    document_repository = FakeDocumentRepository()

    result = asyncio.run(
        RebuildKnowledgeBase().execute(
            artifacts=IngestPipelineArtifacts(
                deduped_docs=[_document()],
                deduped_chunks=[_chunk()],
            ),
            uow=FakeUnitOfWork(document_repository, chunk_repository),
        )
    )

    assert result.status == "completed"
    assert result.document_count == 1
    assert result.chunk_count == 1
    assert result.deleted_document_count == 0
    assert result.deleted_chunk_count == 0
    assert document_repository.deleted_all is True
    assert document_repository.documents == [_document()]
    assert chunk_repository.chunks == [_chunk()]


def test_persist_ingest_artifacts_counts_chunk_tokens_before_postgres_upsert():
    chunk_repository = FakeChunkRepository()
    document_repository = FakeDocumentRepository()
    token_counter = FakeTokenCounter(token_count=4)

    asyncio.run(
        RebuildKnowledgeBase().execute(
            artifacts=IngestPipelineArtifacts(
                deduped_docs=[_document()],
                deduped_chunks=[_chunk()],
            ),
            uow=FakeUnitOfWork(document_repository, chunk_repository),
            token_counter=token_counter,
        )
    )

    assert token_counter.chunks == [_chunk()]
    assert chunk_repository.chunks[0].token_count == 4


def test_persist_ingest_artifacts_refuses_empty_replace_payload():
    with pytest.raises(ValueError, match="zero documents"):
        asyncio.run(
            RebuildKnowledgeBase().execute(
                artifacts=IngestPipelineArtifacts(),
                uow=FakeUnitOfWork(FakeDocumentRepository(), FakeChunkRepository()),
            )
        )


def test_persist_ingest_artifacts_refuses_documents_without_chunks():
    chunk = replace(_chunk(), source_path="std/other.html")

    with pytest.raises(ValueError, match="documents without chunks"):
        asyncio.run(
            RebuildKnowledgeBase().execute(
                artifacts=IngestPipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[chunk],
                ),
                uow=FakeUnitOfWork(FakeDocumentRepository(), FakeChunkRepository()),
            )
        )


def test_persist_ingest_artifacts_refuses_chunks_without_matching_documents():
    chunk = replace(_chunk(), source_path="std/missing.html")

    with pytest.raises(ValueError, match="chunks without matching documents"):
        asyncio.run(
            RebuildKnowledgeBase().execute(
                artifacts=IngestPipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[_chunk(), chunk],
                ),
                uow=FakeUnitOfWork(FakeDocumentRepository(), FakeChunkRepository()),
            )
        )
