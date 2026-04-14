"""FastAPI application entry point for the serving layer."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from rust_assistant.api.routers import chat_router, search_router, system_router
from rust_assistant.clients.llm import StubLLMClient
from rust_assistant.clients.vectordb import StubVectorStoreClient
from rust_assistant.core.config import get_settings
from rust_assistant.core.db import (
    build_async_engine,
    build_session_factory,
    database_is_ready,
    dispose_engine,
)
from rust_assistant.core.logging import configure_logging
from rust_assistant.repositories import ChunkRepository
from rust_assistant.retrieval import DatabaseBackedRetriever, PromptBuilder, QAPipeline
from rust_assistant.services import ReadinessService

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the FastAPI application for the serving API."""

    settings = get_settings()
    configure_logging(logging_settings=settings.logging)
    logger.info("Creating FastAPI application in mode=%s", settings.app.api_mode)

    db_engine = build_async_engine(settings.postgres)
    db_session_factory = build_session_factory(db_engine)
    vector_store_client = StubVectorStoreClient()
    retriever = DatabaseBackedRetriever(
        vector_store=vector_store_client,
        chunk_repository=ChunkRepository(),
    )
    prompt_builder = PromptBuilder()
    llm_client = StubLLMClient()
    qa_pipeline = QAPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm_client,
    )

    dependency_statuses = {
        "postgres": settings.dependencies.postgres,
        "qdrant": settings.dependencies.qdrant,
    }

    async def probe_postgres() -> str:
        if db_session_factory is None:
            return dependency_statuses["postgres"]
        return "ready" if await database_is_ready(db_session_factory) else "not_ready"

    async def probe_qdrant() -> str:
        if not settings.qdrant.url:
            return dependency_statuses["qdrant"]
        try:
            return "ready" if await vector_store_client.ping() else "not_ready"
        except Exception:
            return "not_ready"

    app = FastAPI(
        title="Rust Assistant API",
        version="0.1.0",
    )
    app.state.dependencies = dependency_statuses
    app.state.api_mode = settings.app.api_mode
    app.state.db_engine = db_engine
    app.state.db_session_factory = db_session_factory
    app.state.vector_store_client = vector_store_client
    app.state.retriever = retriever
    app.state.qa_pipeline = qa_pipeline
    app.state.readiness_service = ReadinessService(
        mode=app.state.api_mode,
        dependencies=app.state.dependencies,
        dependency_probes={
            "postgres": probe_postgres,
            "qdrant": probe_qdrant,
        },
    )

    if db_engine is not None:

        @app.on_event("shutdown")
        async def _dispose_db_engine() -> None:
            await dispose_engine(db_engine)

    app.include_router(system_router)
    app.include_router(search_router)
    app.include_router(chat_router)
    return app


app = create_app()
