"""Composition-root helpers that turn global settings into concrete runtime wiring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import httpx
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from rust_assistant.application.services.prompt_builder import PromptBuilder
from rust_assistant.application.services.retrieval.pipeline import RetrievalPipeline
from rust_assistant.application.use_cases.chat import ChatUseCase
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
from rust_assistant.infrastructure.adapters.llm.openai.openai_llm_client import OpenAILLMClient
from rust_assistant.infrastructure.adapters.reranking.tei.tei_reranking_client import (
    TeiRerankingClient,
)
from rust_assistant.infrastructure.adapters.tokenization.tiktoken.tiktoken_tokenizer import (
    TiktokenTokenizer,
)
from rust_assistant.infrastructure.adapters.vector_storage.qdrant.qdrant_vector_storage import (
    QdrantVectorStorage,
)

if TYPE_CHECKING:
    from aiogram import Bot, Dispatcher


class RuntimeConfigurationError(ValueError):
    """Raised when runtime API dependencies cannot be built from settings."""


@dataclass(slots=True, frozen=True)
class RuntimeContainer:
    """Resolved runtime dependencies for current entrypoints."""

    settings: Settings
    sqlalchemy: SqlAlchemyConfig
    search_use_case: Optional[SearchUseCase] = None
    chat_use_case: Optional[ChatUseCase] = None
    db_engine: Optional[AsyncEngine] = None
    http_client: Optional[httpx.AsyncClient] = None
    qdrant_client: Optional[AsyncQdrantClient] = None
    telegram_bot: Optional["Bot"] = None
    telegram_dispatcher: Optional["Dispatcher"] = None

    async def aclose(self) -> None:
        """Close runtime clients owned by the container."""
        if self.telegram_bot is not None:
            await self.telegram_bot.session.close()
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


def _build_reranking_client(
    *,
    settings: Settings,
    http_client: httpx.AsyncClient,
) -> TeiRerankingClient:
    if settings.reranker.base_url is None:
        raise RuntimeConfigurationError(
            "RERANKER_BASE_URL must be configured before serving search requests"
        )
    return TeiRerankingClient(
        client=http_client,
        base_url=settings.reranker.base_url,
        max_batch_items=settings.reranker.max_batch_items,
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


def _build_openai_client(settings: Settings):
    if settings.openai.api_key is None:
        raise RuntimeConfigurationError(
            "OPENAI_API_KEY must be configured before serving chat requests"
        )
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeConfigurationError(
            "openai must be installed before serving chat requests"
        ) from exc

    return AsyncOpenAI(
        api_key=settings.openai.api_key,
        timeout=settings.openai.request_timeout_seconds,
    )


def _build_llm_client(settings: Settings) -> OpenAILLMClient:
    if settings.openai.model is None:
        raise RuntimeConfigurationError(
            "OPENAI_MODEL must be configured before serving chat requests"
        )
    return OpenAILLMClient(
        client=_build_openai_client(settings),
        model=settings.openai.model,
        max_output_tokens=settings.openai.max_output_tokens,
    )


def _build_telegram_runtime(
    *,
    settings: Settings,
    chat_use_case: ChatUseCase,
) -> tuple[Optional["Bot"], Optional["Dispatcher"]]:
    """Build optional aiogram runtime objects for the Telegram webhook entrypoint."""
    if settings.telegram.bot_token is None:
        return None, None
    try:
        from aiogram import Bot, Dispatcher
    except ImportError as exc:
        raise RuntimeConfigurationError(
            "aiogram must be installed before serving Telegram webhook requests"
        ) from exc

    from rust_assistant.infrastructure.entrypoints.webhooks.telegram.handlers import (
        register_telegram_handlers,
    )

    dispatcher = Dispatcher(chat_use_case=chat_use_case)
    register_telegram_handlers(dispatcher)
    return Bot(token=settings.telegram.bot_token), dispatcher


def _build_chat_tokenizer(settings: Settings) -> TiktokenTokenizer:
    if settings.openai.model is None:
        raise RuntimeConfigurationError(
            "OPENAI_MODEL must be configured before serving chat requests"
        )
    try:
        return TiktokenTokenizer(settings.openai.model)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeConfigurationError(str(exc)) from exc


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
    retrieval_pipeline = RetrievalPipeline(
        embedding_client=_build_embedding_client(
            settings=runtime_settings,
            http_client=http_client,
        ),
        vector_storage=_build_vector_storage(
            settings=runtime_settings,
            qdrant_client=qdrant_client,
        ),
        reranking_client=_build_reranking_client(
            settings=runtime_settings,
            http_client=http_client,
        ),
        uow=SqlAlchemyUnitOfWork(session_factory),
    )
    chat_tokenizer = _build_chat_tokenizer(runtime_settings)
    prompt_builder = PromptBuilder(
        tokenizer=chat_tokenizer,
        max_context_tokens=runtime_settings.chat.max_context_tokens,
    )
    search_use_case = SearchUseCase(
        retrieval_pipeline=retrieval_pipeline,
    )
    chat_use_case = ChatUseCase(
        retrieval_pipeline=retrieval_pipeline,
        prompt_builder=prompt_builder,
        llm_client=_build_llm_client(runtime_settings),
        tokenizer=chat_tokenizer,
        retrieval_limit=runtime_settings.chat.retrieval_limit,
        reranking_limit=runtime_settings.chat.reranking_limit,
        use_reranking=runtime_settings.chat.use_reranking,
        max_query_tokens=runtime_settings.chat.max_query_tokens,
    )
    telegram_bot, telegram_dispatcher = _build_telegram_runtime(
        settings=runtime_settings,
        chat_use_case=chat_use_case,
    )
    return RuntimeContainer(
        settings=runtime_settings,
        sqlalchemy=sqlalchemy_config,
        search_use_case=search_use_case,
        chat_use_case=chat_use_case,
        db_engine=db_engine,
        http_client=http_client,
        qdrant_client=qdrant_client,
        telegram_bot=telegram_bot,
        telegram_dispatcher=telegram_dispatcher,
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
