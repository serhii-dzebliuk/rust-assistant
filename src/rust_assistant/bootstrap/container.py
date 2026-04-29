"""Composition-root helpers that turn global settings into concrete runtime wiring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import httpx
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from rust_assistant.application.use_cases.search import SearchUseCase
from rust_assistant.bootstrap.logging import configure_logging
from rust_assistant.bootstrap.settings import LoggingSettings, Settings, get_settings
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.config import SqlAlchemyConfig
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.session import (
    AsyncSessionFactory,
    build_async_engine,
    build_session_factory,
)
from rust_assistant.infrastructure.adapters.data_storage.sqlalchemy.uow import SqlAlchemyUnitOfWork
from rust_assistant.infrastructure.adapters.embedding.tei.tei_embedding_client import (
    TeiEmbeddingClient,
)
from rust_assistant.infrastructure.adapters.vector_storage.qdrant.qdrant_vector_storage import (
    QdrantVectorStorage,
)


class RuntimeConfigurationError(ValueError):
    """Raised when runtime API dependencies cannot be built from settings."""


@dataclass(slots=True, frozen=True)
class RuntimeContainer:
    """Resolved runtime dependencies for current entrypoints."""

    settings: Settings
    sqlalchemy: SqlAlchemyConfig
    search_use_case: Optional[SearchUseCase] = None
    db_engine: Optional[AsyncEngine] = None
    http_client: Optional[httpx.AsyncClient] = None
    qdrant_client: Optional[AsyncQdrantClient] = None

    async def aclose(self) -> None:
        """Close runtime clients owned by the container."""
        if self.http_client is not None:
            await self.http_client.aclose()
        if self.qdrant_client is not None:
            await self.qdrant_client.close()
        if self.db_engine is not None:
            await self.db_engine.dispose()


def _build_sqlalchemy_config(settings: Settings) -> SqlAlchemyConfig:
    """Map global app settings into the SQLAlchemy adapter configuration."""
    postgres = settings.postgres
    return SqlAlchemyConfig(
        url=postgres.url,
        echo=postgres.echo,
        pool_size=postgres.pool_size,
        max_overflow=postgres.max_overflow,
    )


def _require_session_factory(
    session_factory: Optional[AsyncSessionFactory],
) -> AsyncSessionFactory:
    if session_factory is None:
        raise RuntimeConfigurationError(
            "DATABASE_URL must be configured before serving search requests"
        )
    return session_factory


def _build_http_client(settings: Settings) -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        settings.embedding.request_timeout_seconds,
        connect=10.0,
    )
    return httpx.AsyncClient(timeout=timeout)


def _build_embedding_client(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
) -> TeiEmbeddingClient:
    provider = settings.embedding.provider
    if provider != "tei":
        raise RuntimeConfigurationError(
            "EMBEDDING_PROVIDER must be configured as 'tei' before serving search requests"
        )
    if settings.embedding.base_url is None:
        raise RuntimeConfigurationError(
            "EMBEDDING_BASE_URL must be configured before serving search requests"
        )
    return TeiEmbeddingClient(
        client=http_client,
        base_url=settings.embedding.base_url,
        normalize=settings.embedding.normalize,
        max_batch_tokens=settings.embedding.max_batch_tokens,
        max_batch_items=settings.embedding.max_batch_items,
    )


def _build_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    if settings.qdrant.url is None:
        raise RuntimeConfigurationError(
            "QDRANT_URL must be configured before serving search requests"
        )
    return AsyncQdrantClient(url=settings.qdrant.url)


def _build_vector_storage(
    *,
    settings: Settings,
    qdrant_client: AsyncQdrantClient,
) -> QdrantVectorStorage:
    if settings.qdrant.vector_size is None:
        raise RuntimeConfigurationError(
            "QDRANT_VECTOR_SIZE must be configured before serving search requests"
        )
    return QdrantVectorStorage(
        client=qdrant_client,
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
        distance=settings.qdrant.distance,
        upsert_batch_size=settings.qdrant.upsert_batch_size,
    )


def build_container(
    *,
    settings: Optional[Settings] = None,
    logging_settings: Optional[LoggingSettings] = None,
    include_search: bool = False,
) -> RuntimeContainer:
    """Load runtime settings, configure logging, and wire adapter configuration."""
    runtime_settings = settings or get_settings()
    configure_logging(logging_settings=logging_settings or runtime_settings.logging)
    sqlalchemy_config = _build_sqlalchemy_config(runtime_settings)
    if not include_search:
        return RuntimeContainer(
            settings=runtime_settings,
            sqlalchemy=sqlalchemy_config,
        )

    db_engine = build_async_engine(sqlalchemy_config)
    session_factory = _require_session_factory(build_session_factory(db_engine))
    http_client = _build_http_client(runtime_settings)
    qdrant_client = _build_qdrant_client(runtime_settings)
    return RuntimeContainer(
        settings=runtime_settings,
        sqlalchemy=sqlalchemy_config,
        search_use_case=SearchUseCase(
            embedding_client=_build_embedding_client(
                settings=runtime_settings,
                http_client=http_client,
            ),
            vector_storage=_build_vector_storage(
                settings=runtime_settings,
                qdrant_client=qdrant_client,
            ),
            uow=SqlAlchemyUnitOfWork(session_factory),
        ),
        db_engine=db_engine,
        http_client=http_client,
        qdrant_client=qdrant_client,
    )


def build_container_with_log_level(
    *,
    settings: Optional[Settings] = None,
    log_level: Optional[str] = None,
    include_search: bool = False,
) -> RuntimeContainer:
    """Build the runtime container with an optional temporary log-level override."""
    runtime_settings = settings or get_settings()
    logging_settings = runtime_settings.logging
    if log_level is not None:
        logging_settings = replace(logging_settings, level=log_level)
    return build_container(
        settings=runtime_settings,
        logging_settings=logging_settings,
        include_search=include_search,
    )
