import asyncio
from dataclasses import replace

import pytest

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.ports.vector_storage import VectorPayload, VectorPoint
from rust_assistant.application.use_cases.ingest.rebuild_knowledge_base import (
    RebuildKnowledgeBaseCommand,
    RebuildKnowledgeBaseUseCase,
)
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate, ItemType

pytestmark = pytest.mark.unit


class FakeDocumentRepository:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.deleted_all = False
        self.documents = None

    async def delete_all(self):
        self.events.append("documents.delete_all")
        self.deleted_all = True

    async def add_many(self, documents):
        self.events.append("documents.add_many")
        self.documents = list(documents)


class FakeChunkRepository:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.chunks = None

    async def add_many(self, chunks):
        self.events.append("chunks.add_many")
        self.chunks = list(chunks)


class FakeUnitOfWork:
    def __init__(self, document_repository, chunk_repository, events=None, fail_commit=False):
        self.events = events if events is not None else []
        self.documents = document_repository
        self.chunks = chunk_repository
        self.committed = False
        self.fail_commit = fail_commit

    async def __aenter__(self):
        self.events.append("uow.enter")
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.events.append("uow.exit")
        return False

    async def commit(self):
        self.events.append("uow.commit")
        if self.fail_commit:
            raise RuntimeError("commit failed")
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


class FakeEmbeddingClient:
    def __init__(self, vectors=None, events=None, fail=False):
        self.vectors = vectors if vectors is not None else [[0.1, 0.2, 0.3]]
        self.events = events if events is not None else []
        self.inputs = []
        self.fail = fail

    async def embed_text(self, text):
        return self.vectors[0]

    async def embed_texts(self, inputs):
        self.events.append("embedding.embed_texts")
        self.inputs = list(inputs)
        if self.fail:
            raise RuntimeError("embedding failed")
        return self.vectors


class FakeVectorStorage:
    def __init__(self, events=None, fail_recreate=False):
        self.events = events if events is not None else []
        self.fail_recreate = fail_recreate
        self.recreated = False
        self.points = None

    async def recreate_collection(self):
        self.events.append("qdrant.recreate_collection")
        if self.fail_recreate:
            raise RuntimeError("qdrant failed")
        self.recreated = True

    async def upsert_vectors(self, points):
        self.events.append("qdrant.upsert_vectors")
        self.points = list(points)

    async def search(self, query_vector, limit, score_threshold=None):
        return []


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


def _use_case(
    *,
    uow=None,
    tokenizer=None,
    embedding_client=None,
    vector_storage=None,
):
    return RebuildKnowledgeBaseUseCase(
        uow=uow or FakeUnitOfWork(FakeDocumentRepository(), FakeChunkRepository()),
        tokenizer=tokenizer or FakeTokenizer(token_count=3),
        embedding_client=embedding_client or FakeEmbeddingClient(),
        vector_storage=vector_storage or FakeVectorStorage(),
    )


def test_persist_ingest_artifacts_persists_documents_and_chunks_to_postgres():
    chunk_repository = FakeChunkRepository()
    document_repository = FakeDocumentRepository()
    tokenizer = FakeTokenizer(token_count=3)
    embedding_client = FakeEmbeddingClient(vectors=[[0.1, 0.2, 0.3]])
    vector_storage = FakeVectorStorage()
    uow = FakeUnitOfWork(document_repository, chunk_repository)

    result = asyncio.run(
        RebuildKnowledgeBaseUseCase(
            uow=uow,
            tokenizer=tokenizer,
            embedding_client=embedding_client,
            vector_storage=vector_storage,
        ).execute(
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
    assert result.vector_count == 1
    assert result.vector_status == "synced"
    assert result.deleted_document_count == 0
    assert result.deleted_chunk_count == 0
    assert document_repository.deleted_all is True
    assert document_repository.documents == [_document()]
    assert chunk_repository.chunks == [replace(_chunk(), token_count=3)]
    assert vector_storage.recreated is True
    assert len(vector_storage.points) == 1
    assert vector_storage.points[0] == VectorPoint(
        chunk_id=_chunk().id,
        vector=[0.1, 0.2, 0.3],
        payload=VectorPayload(
            document_id=_chunk().document_id,
            crate="std",
            item_type="unknown",
            source_path="std/keyword.async.html",
            item_path="std::keyword::async",
            rust_version="1.91.1",
            section_title="std::keyword::async",
            chunk_index=0,
            text_hash=_chunk().text_hash,
        ),
    )


def test_persist_ingest_artifacts_counts_chunk_tokens_before_postgres_upsert():
    chunk_repository = FakeChunkRepository()
    document_repository = FakeDocumentRepository()
    tokenizer = FakeTokenizer(token_count=4)
    embedding_client = FakeEmbeddingClient()
    uow = FakeUnitOfWork(document_repository, chunk_repository)

    asyncio.run(
        RebuildKnowledgeBaseUseCase(
            uow=uow,
            tokenizer=tokenizer,
            embedding_client=embedding_client,
            vector_storage=FakeVectorStorage(),
        ).execute(
            RebuildKnowledgeBaseCommand(
                artifacts=IngestPipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[_chunk()],
                ),
            )
        )
    )

    assert tokenizer.texts == ["Returns a Future."]
    assert embedding_client.inputs[0].text == "Returns a Future."
    assert embedding_client.inputs[0].token_count == 4
    assert chunk_repository.chunks[0].token_count == 4


def test_persist_ingest_artifacts_embeds_before_database_writes_and_syncs_qdrant_after_commit():
    events = []
    chunk_repository = FakeChunkRepository(events)
    document_repository = FakeDocumentRepository(events)
    uow = FakeUnitOfWork(document_repository, chunk_repository, events)
    embedding_client = FakeEmbeddingClient(events=events)
    vector_storage = FakeVectorStorage(events=events)

    asyncio.run(
        _use_case(
            uow=uow,
            embedding_client=embedding_client,
            vector_storage=vector_storage,
        ).execute(
            RebuildKnowledgeBaseCommand(
                artifacts=IngestPipelineArtifacts(
                    deduped_docs=[_document()],
                    deduped_chunks=[_chunk()],
                ),
            )
        )
    )

    assert events == [
        "embedding.embed_texts",
        "uow.enter",
        "documents.delete_all",
        "documents.add_many",
        "chunks.add_many",
        "uow.commit",
        "uow.exit",
        "qdrant.recreate_collection",
        "qdrant.upsert_vectors",
    ]


def test_persist_ingest_artifacts_does_not_write_postgres_when_embedding_fails():
    events = []

    with pytest.raises(RuntimeError, match="embedding failed"):
        asyncio.run(
            _use_case(
                uow=FakeUnitOfWork(
                    FakeDocumentRepository(events),
                    FakeChunkRepository(events),
                    events,
                ),
                embedding_client=FakeEmbeddingClient(events=events, fail=True),
                vector_storage=FakeVectorStorage(events=events),
            ).execute(
                RebuildKnowledgeBaseCommand(
                    artifacts=IngestPipelineArtifacts(
                        deduped_docs=[_document()],
                        deduped_chunks=[_chunk()],
                    ),
                )
            )
        )

    assert events == ["embedding.embed_texts"]


def test_persist_ingest_artifacts_does_not_recreate_qdrant_when_postgres_commit_fails():
    events = []

    with pytest.raises(RuntimeError, match="commit failed"):
        asyncio.run(
            _use_case(
                uow=FakeUnitOfWork(
                    FakeDocumentRepository(events),
                    FakeChunkRepository(events),
                    events,
                    fail_commit=True,
                ),
                embedding_client=FakeEmbeddingClient(events=events),
                vector_storage=FakeVectorStorage(events=events),
            ).execute(
                RebuildKnowledgeBaseCommand(
                    artifacts=IngestPipelineArtifacts(
                        deduped_docs=[_document()],
                        deduped_chunks=[_chunk()],
                    ),
                )
            )
        )

    assert "qdrant.recreate_collection" not in events


def test_persist_ingest_artifacts_propagates_qdrant_failure_after_postgres_commit():
    events = []

    with pytest.raises(RuntimeError, match="qdrant failed"):
        asyncio.run(
            _use_case(
                uow=FakeUnitOfWork(
                    FakeDocumentRepository(events),
                    FakeChunkRepository(events),
                    events,
                ),
                embedding_client=FakeEmbeddingClient(events=events),
                vector_storage=FakeVectorStorage(events=events, fail_recreate=True),
            ).execute(
                RebuildKnowledgeBaseCommand(
                    artifacts=IngestPipelineArtifacts(
                        deduped_docs=[_document()],
                        deduped_chunks=[_chunk()],
                    ),
                )
            )
        )

    assert "uow.commit" in events
    assert events[-1] == "qdrant.recreate_collection"


def test_persist_ingest_artifacts_refuses_empty_replace_payload():
    with pytest.raises(ValueError, match="zero documents"):
        asyncio.run(
            _use_case(
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
            _use_case(
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
            _use_case(
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
