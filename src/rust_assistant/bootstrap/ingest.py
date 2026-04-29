"""Ingest runtime wiring and orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import httpx
from qdrant_client import AsyncQdrantClient

from rust_assistant.application.dto.ingest_pipeline import IngestPipelineArtifacts
from rust_assistant.application.use_cases.ingest.discover_documents import (
    DiscoverDocumentsUseCase,
)
from rust_assistant.application.use_cases.ingest.parse_documents import ParseDocumentsUseCase
from rust_assistant.application.use_cases.ingest.rebuild_knowledge_base import (
    RebuildKnowledgeBaseCommand,
    RebuildKnowledgeBaseResult,
    RebuildKnowledgeBaseUseCase,
)
from rust_assistant.application.use_cases.ingest.ingest_documents import (
    IngestDocumentsCommand,
    IngestDocumentsUseCase,
)
from rust_assistant.bootstrap.container import build_container_with_log_level
from rust_assistant.bootstrap.settings import Settings
from rust_assistant.domain.enums import Crate
from rust_assistant.infrastructure.adapters.parsing.html.document_parser import (
    HtmlDocumentParser,
)
from rust_assistant.infrastructure.adapters.data_source.filesystem.document_discoverer import (
    RawDocsDocumentDiscoverer,
)
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.config import SqlAlchemyConfig
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.session import (
    build_async_engine,
    build_session_factory,
    database_is_ready,
    dispose_engine,
)
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.uow import SqlAlchemyUnitOfWork
from rust_assistant.infrastructure.adapters.embedding.tei.tei_embedding_client import (
    TeiEmbeddingClient,
)
from rust_assistant.infrastructure.adapters.tokenization.transformers.transformers_tokenizer import (
    TransformersTokenizer,
)
from rust_assistant.infrastructure.adapters.vector_storage.qdrant.qdrant_vector_storage import (
    QdrantVectorStorage,
)

logger = logging.getLogger(__name__)

SUPPORTED_CRATES = (Crate.STD, Crate.BOOK, Crate.CARGO, Crate.REFERENCE)
PERSISTABLE_STAGES = ("chunk_dedup", "all")
IngestPersistenceResult = RebuildKnowledgeBaseResult


class IngestDatabaseUnavailableError(RuntimeError):
    """Raised when the ingest runtime cannot reach PostgreSQL."""


class IngestConfigurationError(ValueError):
    """Raised when persisted ingest is missing required configuration."""


class IngestTokenizerUnavailableError(RuntimeError):
    """Raised when the embedding tokenizer cannot be loaded for persisted ingest."""


def _resolve_raw_docs_dir(settings: Settings) -> Path:
    """Resolve and validate the raw Rust documentation directory for ingest."""
    raw_docs_dir = settings.ingest.raw_docs_dir
    if raw_docs_dir is None:
        raise ValueError("RUST_DOCS_RAW_DIR must be configured before running ingest")

    resolved = raw_docs_dir.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"RUST_DOCS_RAW_DIR does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"RUST_DOCS_RAW_DIR must point to a directory: {resolved}")
    return resolved


def _selected_crates(crates: Optional[Sequence[str]]) -> list[Crate]:
    """Return the crate scope selected for this ingest run."""
    selected_names = list(dict.fromkeys(crates or [crate.value for crate in SUPPORTED_CRATES]))
    selected_crates: list[Crate] = []
    for crate_name in selected_names:
        try:
            selected_crates.append(Crate(crate_name))
        except ValueError as exc:
            raise ValueError(f"Unsupported crate: {crate_name}") from exc
    return selected_crates


def _log_stage_summary(stage: str, artifacts: IngestPipelineArtifacts) -> None:
    """Log a concise summary for the completed ingest stage."""
    logger.info("Discovered files: %s", len(artifacts.discovered_files))
    if stage == "discover":
        return

    logger.info("Parsed documents: %s", len(artifacts.parsed_docs))
    if stage == "parse":
        return

    logger.info("Cleaned documents: %s", len(artifacts.cleaned_docs))
    if stage == "clean":
        return

    logger.info("Deduplicated documents: %s", len(artifacts.deduped_docs))
    if stage == "dedup":
        return

    logger.info("Generated chunks: %s", len(artifacts.chunks))
    if stage == "chunk":
        return

    logger.info("Deduplicated chunks: %s", len(artifacts.deduped_chunks))


def _validate_options(
    *,
    stage: str,
    persist: bool,
    limit: Optional[int],
    allow_sample_persist: bool,
) -> None:
    """Validate combinations that are unsafe for PostgreSQL replacement."""
    if persist and stage not in PERSISTABLE_STAGES:
        raise ValueError("PostgreSQL persistence requires --stage chunk_dedup or --stage all")
    if persist and limit is not None and not allow_sample_persist:
        raise ValueError(
            "--limit with persistence replaces PostgreSQL and Qdrant with a sample; "
            "pass --allow-sample-persist to confirm"
        )


def _build_tokenizer(settings: Settings) -> TransformersTokenizer:
    """Build the mandatory embedding-model tokenizer for persisted ingest."""
    embedding_model = settings.embedding.model
    if embedding_model is None:
        raise IngestConfigurationError(
            "EMBEDDING_MODEL must be configured before persisted ingest can count chunk tokens"
        )

    try:
        return TransformersTokenizer(embedding_model)
    except Exception as exc:
        raise IngestTokenizerUnavailableError(
            "Could not load Hugging Face tokenizer for EMBEDDING_MODEL "
            f"{embedding_model!r}; persisted ingest requires token_count values"
        ) from exc


def _build_embedding_client(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
) -> TeiEmbeddingClient:
    """Build the configured embedding client for persisted ingest."""
    provider = settings.embedding.provider
    if provider != "tei":
        raise IngestConfigurationError(
            "EMBEDDING_PROVIDER must be configured as 'tei' before persisted ingest"
        )
    if settings.embedding.base_url is None:
        raise IngestConfigurationError(
            "EMBEDDING_BASE_URL must be configured before persisted ingest can embed chunks"
        )

    return TeiEmbeddingClient(
        client=http_client,
        base_url=settings.embedding.base_url,
        normalize=settings.embedding.normalize,
        max_batch_tokens=settings.embedding.max_batch_tokens,
        max_batch_items=settings.embedding.max_batch_items,
    )


def _build_vector_storage(settings: Settings) -> QdrantVectorStorage:
    """Build the configured Qdrant vector storage adapter."""
    if settings.qdrant.url is None:
        raise IngestConfigurationError(
            "QDRANT_URL must be configured before persisted ingest can sync vectors"
        )
    if settings.qdrant.vector_size is None:
        raise IngestConfigurationError(
            "QDRANT_VECTOR_SIZE must be configured before persisted ingest can sync vectors"
        )

    return QdrantVectorStorage(
        client=AsyncQdrantClient(url=settings.qdrant.url),
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
        distance=settings.qdrant.distance,
        upsert_batch_size=settings.qdrant.upsert_batch_size,
    )


def _build_pipeline(*, raw_docs_dir: Path) -> IngestDocumentsUseCase:
    """Build the ingest pipeline use case with filesystem-backed adapters."""
    return IngestDocumentsUseCase(
        discover_documents=DiscoverDocumentsUseCase(RawDocsDocumentDiscoverer(raw_docs_dir)),
        parse_documents=ParseDocumentsUseCase(HtmlDocumentParser(raw_docs_dir)),
    )


def _run_pipeline_artifacts(
    *,
    raw_docs_dir: Path,
    stage: str,
    crates: list[Crate],
    limit: Optional[int],
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> IngestPipelineArtifacts:
    """Execute the ingest application pipeline and return all stage artifacts."""
    result = _build_pipeline(raw_docs_dir=raw_docs_dir).execute(
        IngestDocumentsCommand(
            stage=stage,
            crates=crates,
            limit=limit,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )
    )
    return result.artifacts


async def _persist_after_pipeline(
    *,
    sqlalchemy_config: SqlAlchemyConfig,
    settings: Settings,
    artifacts: IngestPipelineArtifacts,
) -> IngestPersistenceResult:
    """Persist completed pipeline artifacts using one async database lifecycle."""
    db_engine = build_async_engine(sqlalchemy_config)
    session_factory = build_session_factory(db_engine)
    try:
        if session_factory is None or not await database_is_ready(session_factory):
            raise IngestDatabaseUnavailableError(
                "DATABASE_URL must point to a reachable PostgreSQL database"
            )
        tokenizer = _build_tokenizer(settings)
        vector_storage = _build_vector_storage(settings)
        timeout = httpx.Timeout(
            settings.embedding.request_timeout_seconds,
            connect=10.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            embedding_client = _build_embedding_client(
                settings=settings,
                http_client=http_client,
            )
            return await RebuildKnowledgeBaseUseCase(
                uow=SqlAlchemyUnitOfWork(session_factory),
                tokenizer=tokenizer,
                embedding_client=embedding_client,
                vector_storage=vector_storage,
            ).execute(
                RebuildKnowledgeBaseCommand(
                    artifacts=artifacts,
                )
            )
    finally:
        await dispose_engine(db_engine)


def run_ingest(
    *,
    stage: str = "all",
    crates: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    persist: bool = True,
    allow_sample_persist: bool = False,
    verbose: bool = False,
) -> int:
    """Run the ingest pipeline using bootstrap-managed runtime wiring."""
    _validate_options(
        stage=stage,
        persist=persist,
        limit=limit,
        allow_sample_persist=allow_sample_persist,
    )

    log_level = "DEBUG" if verbose else None
    container = build_container_with_log_level(log_level=log_level)
    settings = container.settings
    raw_docs_dir = _resolve_raw_docs_dir(settings)
    selected_crates = _selected_crates(crates)
    logger.info(
        "Starting ingest pipeline stage=%s crates=%s persist_postgres=%s limit=%s "
        "allow_sample_persist=%s",
        stage,
        [crate.value for crate in selected_crates],
        persist,
        limit,
        allow_sample_persist,
    )

    artifacts = _run_pipeline_artifacts(
        raw_docs_dir=raw_docs_dir,
        stage=stage,
        crates=selected_crates,
        limit=limit,
        max_chunk_chars=settings.ingest.max_chunk_chars,
        min_chunk_chars=settings.ingest.min_chunk_chars,
    )
    _log_stage_summary(stage, artifacts)

    if not persist:
        return 0

    persistence_result = asyncio.run(
        _persist_after_pipeline(
            sqlalchemy_config=container.sqlalchemy,
            settings=settings,
            artifacts=artifacts,
        )
    )
    logger.info(
        "Persisted ingest status=%s docs=%s chunks=%s vectors=%s vector_status=%s "
        "deleted_docs=%s deleted_chunks=%s",
        persistence_result.status,
        persistence_result.document_count,
        persistence_result.chunk_count,
        persistence_result.vector_count,
        persistence_result.vector_status,
        persistence_result.deleted_document_count,
        persistence_result.deleted_chunk_count,
    )
    return 0
