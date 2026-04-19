import os

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from rust_assistant.core.config import get_settings
from rust_assistant.core.db import (
    build_async_engine,
    build_session_factory,
    database_is_ready,
    dispose_engine,
)
from rust_assistant.ingest import run as ingest_run
from rust_assistant.ingest.entities import Chunk, ChunkMetadata, Document, DocumentMetadata
from rust_assistant.ingest.persist import IngestPersistenceResult, persist_ingest_artifacts
from rust_assistant.ingest.pipeline import PipelineArtifacts
from rust_assistant.models import ChunkRecord, DocumentRecord
from rust_assistant.schemas.enums import Crate, ItemType

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def db_settings():
    if os.getenv("RUN_DB_INTEGRATION") != "1":
        pytest.skip("Set RUN_DB_INTEGRATION=1 to run real PostgreSQL integration tests")

    get_settings.cache_clear()
    settings = get_settings()
    if settings.postgres.url is None:
        pytest.skip("DATABASE_URL is required for PostgreSQL integration tests")
    return settings


def _document(source_path: str = "unknown/integration-test.html") -> Document:
    return Document(
        doc_id=Document.generate_id(source_path, "Integration test"),
        title="Integration test",
        source_path=source_path,
        text="Integration test document body.",
        metadata=DocumentMetadata(
            crate=Crate.UNKNOWN,
            item_path="unknown::integration-test",
            item_type=ItemType.UNKNOWN,
            rust_version=None,
            url="https://example.invalid/integration-test.html",
        ),
    )


def _chunk(document: Document) -> Chunk:
    return Chunk(
        chunk_id=Chunk.generate_id(document.doc_id, 0),
        doc_id=document.doc_id,
        text="Integration test document body.",
        metadata=ChunkMetadata(
            crate=Crate.UNKNOWN,
            item_path=document.metadata.item_path,
            item_type=document.metadata.item_type,
            rust_version=document.metadata.rust_version,
            url=document.metadata.url,
            section="Integration test",
            section_path=["Integration test"],
            anchor="integration-test",
            chunk_index=0,
            start_char=0,
            end_char=len(document.text),
            doc_title=document.title,
            doc_source_path=document.source_path,
        ),
    )


def _artifacts(source_path: str = "unknown/integration-test.html") -> PipelineArtifacts:
    document = _document(source_path=source_path)
    return PipelineArtifacts(deduped_docs=[document], deduped_chunks=[_chunk(document)])


async def test_database_is_ready_with_real_postgres(db_settings):
    engine = build_async_engine(db_settings.postgres)
    session_factory = build_session_factory(engine)
    try:
        assert await database_is_ready(session_factory) is True
    finally:
        await dispose_engine(engine)


async def test_persist_ingest_artifacts_writes_documents_and_chunks_then_rolls_back(
    db_settings,
):
    engine = build_async_engine(db_settings.postgres)
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
                result = await persist_ingest_artifacts(
                    artifacts=_artifacts(source_path=source_path),
                    session_factory=session_factory,
                    replace_crates=[Crate.UNKNOWN.value],
                )

                document_count = await connection.scalar(
                    select(func.count(DocumentRecord.id)).where(
                        DocumentRecord.source_path == source_path
                    )
                )
                chunk_count = await connection.scalar(
                    select(func.count(ChunkRecord.id))
                    .join(DocumentRecord)
                    .where(DocumentRecord.source_path == source_path)
                )

                assert result.document_count == 1
                assert result.chunk_count == 1
                assert document_count == 1
                assert chunk_count == 1
            finally:
                await outer_transaction.rollback()
    finally:
        await dispose_engine(engine)


async def test_cli_db_helper_uses_single_event_loop(db_settings, monkeypatch):
    artifacts = PipelineArtifacts()

    async def fake_persist_ingest_artifacts(**kwargs):
        assert kwargs["artifacts"] is artifacts
        assert kwargs["replace_crates"] == ["std"]
        assert await database_is_ready(kwargs["session_factory"]) is True
        return IngestPersistenceResult(status="completed", document_count=0, chunk_count=0)

    monkeypatch.setattr(
        "rust_assistant.ingest.run.persist_ingest_artifacts",
        fake_persist_ingest_artifacts,
    )

    result = await ingest_run._persist_after_pipeline(
        settings=db_settings,
        artifacts=artifacts,
        selected_crates=["std"],
    )

    assert result.status == "completed"
