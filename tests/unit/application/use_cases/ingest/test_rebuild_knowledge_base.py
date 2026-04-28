import asyncio
from dataclasses import replace

import pytest

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.rebuild_knowledge_base import (
    RebuildKnowledgeBaseCommand,
    RebuildKnowledgeBaseUseCase,
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


class FakeTokenizer:
    def __init__(self, token_count):
        self.token_count = token_count
        self.texts = []

    def count_tokens(self, text):
        self.texts.append(text)
        return self.token_count


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
    tokenizer = FakeTokenizer(token_count=3)
    uow = FakeUnitOfWork(document_repository, chunk_repository)

    result = asyncio.run(
        RebuildKnowledgeBaseUseCase(uow=uow, tokenizer=tokenizer).execute(
            RebuildKnowledgeBaseCommand(
                artifacts=IngestPipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[_chunk()],
                ),
            )
        )
    )

    assert result.status == "completed"
    assert result.document_count == 1
    assert result.chunk_count == 1
    assert result.deleted_document_count == 0
    assert result.deleted_chunk_count == 0
    assert document_repository.deleted_all is True
    assert document_repository.documents == [_document()]
    assert chunk_repository.chunks == [replace(_chunk(), token_count=3)]


def test_persist_ingest_artifacts_counts_chunk_tokens_before_postgres_upsert():
    chunk_repository = FakeChunkRepository()
    document_repository = FakeDocumentRepository()
    tokenizer = FakeTokenizer(token_count=4)
    uow = FakeUnitOfWork(document_repository, chunk_repository)

    asyncio.run(
        RebuildKnowledgeBaseUseCase(uow=uow, tokenizer=tokenizer).execute(
            RebuildKnowledgeBaseCommand(
                artifacts=IngestPipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[_chunk()],
                ),
            )
        )
    )

    assert tokenizer.texts == ["Returns a Future."]
    assert chunk_repository.chunks[0].token_count == 4


def test_persist_ingest_artifacts_refuses_empty_replace_payload():
    with pytest.raises(ValueError, match="zero documents"):
        asyncio.run(
            RebuildKnowledgeBaseUseCase(
                uow=FakeUnitOfWork(FakeDocumentRepository(), FakeChunkRepository()),
                tokenizer=FakeTokenizer(token_count=1),
            ).execute(
                RebuildKnowledgeBaseCommand(
                    artifacts=IngestPipelineArtifacts(),
                )
            )
        )


def test_persist_ingest_artifacts_refuses_documents_without_chunks():
    chunk = replace(_chunk(), source_path="std/other.html")

    with pytest.raises(ValueError, match="documents without chunks"):
        asyncio.run(
            RebuildKnowledgeBaseUseCase(
                uow=FakeUnitOfWork(FakeDocumentRepository(), FakeChunkRepository()),
                tokenizer=FakeTokenizer(token_count=1),
            ).execute(
                RebuildKnowledgeBaseCommand(
                    artifacts=IngestPipelineArtifacts(
                        deduped_docs=[_document()],
                        deduped_chunks=[chunk],
                    ),
                )
            )
        )


def test_persist_ingest_artifacts_refuses_chunks_without_matching_documents():
    chunk = replace(_chunk(), source_path="std/missing.html")

    with pytest.raises(ValueError, match="chunks without matching documents"):
        asyncio.run(
            RebuildKnowledgeBaseUseCase(
                uow=FakeUnitOfWork(FakeDocumentRepository(), FakeChunkRepository()),
                tokenizer=FakeTokenizer(token_count=1),
            ).execute(
                RebuildKnowledgeBaseCommand(
                    artifacts=IngestPipelineArtifacts(
                        deduped_docs=[_document()],
                        deduped_chunks=[_chunk(), chunk],
                    ),
                )
            )
        )
