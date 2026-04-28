import os

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.rebuild_knowledge_base import (
    RebuildKnowledgeBaseCommand,
    RebuildKnowledgeBaseUseCase,
    RebuildKnowledgeBaseResult,
)
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.config import SqlAlchemyConfig
from rust_assistant.bootstrap.ingest import _persist_after_pipeline
from rust_assistant.bootstrap.settings import get_settings
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.session import (
    build_async_engine,
    build_session_factory,
    database_is_ready,
    dispose_engine,
)
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.models import (
    ChunkRecord,
    DocumentRecord,
)
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.uow import SqlAlchemyUnitOfWork

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _sqlalchemy_config(settings) -> SqlAlchemyConfig:
    return SqlAlchemyConfig(
        url=settings.postgres.url,
        echo=settings.postgres.echo,
        pool_size=settings.postgres.pool_size,
        max_overflow=settings.postgres.max_overflow,
    )


@pytest.fixture
def db_settings():
    if os.getenv("RUN_DB_INTEGRATION") != "1":
        pytest.skip("Set RUN_DB_INTEGRATION=1 to run real PostgreSQL integration tests")

    get_settings.cache_clear()
    settings = get_settings()
    if settings.postgres.url is None:
        pytest.skip("DATABASE_URL is required for PostgreSQL integration tests")
    if settings.embedding.model is None:
        pytest.skip("EMBEDDING_MODEL is required for persisted ingest integration tests")
    return settings


def _document(source_path: str = "unknown/integration-test.html") -> Document:
    return Document(
        source_path=source_path,
        title="Integration test",
        text="Integration test document body.",
        crate=Crate.UNKNOWN,
        url="https://example.invalid/integration-test.html",
        item_path="unknown::integration-test",
        item_type=ItemType.UNKNOWN,
        rust_version=None,
    )


def _chunk(document: Document) -> Chunk:
    return Chunk(
        source_path=document.source_path,
        chunk_index=0,
        text="Integration test document body.",
        crate=Crate.UNKNOWN,
        start_offset=0,
        end_offset=len(document.text),
        item_path=document.item_path,
        item_type=document.item_type,
        rust_version=document.rust_version,
        url=document.url,
        section_path=["Integration test"],
        section_anchor="integration-test",
    )


def _artifacts(source_path: str = "unknown/integration-test.html") -> IngestPipelineArtifacts:
    document = _document(source_path=source_path)
    return IngestPipelineArtifacts(deduped_docs=[document], deduped_chunks=[_chunk(document)])


class FakeTokenizer:
    def __init__(self, token_count: int):
        self.token_count = token_count

    def count_tokens(self, _text: str) -> int:
        return self.token_count


async def test_database_is_ready_with_real_postgres(db_settings):
    engine = build_async_engine(_sqlalchemy_config(db_settings))
    session_factory = build_session_factory(engine)
    try:
        assert await database_is_ready(session_factory) is True
    finally:
        await dispose_engine(engine)


async def test_persist_ingest_artifacts_writes_documents_and_chunks_then_rolls_back(
    db_settings,
):
    engine = build_async_engine(_sqlalchemy_config(db_settings))
    source_path = "unknown/integration-rollback.html"
    try:
        async with engine.connect() as connection:
            outer_transaction = await connection.begin()
            session_factory = async_sessionmaker(
                connection,
                expire_on_commit=False,
                autoflush=False,
                join_transaction_mode="create_savepoint",
            )
            try:
                result = await RebuildKnowledgeBaseUseCase(
                    uow=SqlAlchemyUnitOfWork(session_factory),
                    tokenizer=FakeTokenizer(token_count=5),
                ).execute(
                    RebuildKnowledgeBaseCommand(
                        artifacts=_artifacts(source_path=source_path),
                    )
                )

                document_count = await connection.scalar(
                    select(func.count(DocumentRecord.pk)).where(
                        DocumentRecord.source_path == source_path
                    )
                )
                chunk_count = await connection.scalar(
                    select(func.count(ChunkRecord.pk))
                    .join(DocumentRecord)
                    .where(DocumentRecord.source_path == source_path)
                )
                stored_token_count = await connection.scalar(
                    select(ChunkRecord.token_count)
                    .join(DocumentRecord)
                    .where(DocumentRecord.source_path == source_path)
                )

                assert result.document_count == 1
                assert result.chunk_count == 1
                assert document_count == 1
                assert chunk_count == 1
                assert stored_token_count == 5
            finally:
                await outer_transaction.rollback()
    finally:
        await dispose_engine(engine)


async def test_cli_db_helper_uses_single_event_loop(db_settings, monkeypatch):
    expected_artifacts = IngestPipelineArtifacts(
        deduped_docs=[_document()],
        deduped_chunks=[_chunk(_document())],
    )

    class FakeTokenizerAdapter:
        def __init__(self, model_name):
            self.model_name = model_name

    class FakeRebuildKnowledgeBase:
        def __init__(self, *, uow, tokenizer):
            self.uow = uow
            self.tokenizer = tokenizer

        async def execute(self, command):
            assert command.artifacts is expected_artifacts
            assert self.tokenizer.model_name == db_settings.embedding.model
            assert await database_is_ready(self.uow._session_factory) is True
            return RebuildKnowledgeBaseResult(status="completed", document_count=0, chunk_count=0)

    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.RebuildKnowledgeBaseUseCase",
        FakeRebuildKnowledgeBase,
    )
    monkeypatch.setattr(
        "rust_assistant.bootstrap.ingest.TransformersTokenizer",
        FakeTokenizerAdapter,
    )

    result = await _persist_after_pipeline(
        sqlalchemy_config=_sqlalchemy_config(db_settings),
        settings=db_settings,
        artifacts=expected_artifacts,
    )

    assert result.status == "completed"
