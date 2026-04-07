"""FastAPI application entry point for the serving layer."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from rust_assistant.api.routers import chat_router, search_router, system_router
from rust_assistant.clients.llm import StubLLMClient
from rust_assistant.core.config import get_settings
from rust_assistant.core.logging import configure_logging
from rust_assistant.retrieval import PromptBuilder, QAPipeline, StubRetriever
from rust_assistant.services import ChatService, ReadinessService, SearchService

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the FastAPI application for the serving API."""
    settings = get_settings()
    configure_logging(logging_settings=settings.logging)
    logger.info("Creating FastAPI application in mode=%s", settings.app.api_mode)

    app = FastAPI(
        title="Rust Assistant API",
        version="0.1.0",
    )
    app.state.dependencies = {
        "postgres": settings.dependencies.postgres,
        "qdrant": settings.dependencies.qdrant,
    }
    app.state.api_mode = settings.app.api_mode

    retriever = StubRetriever()
    prompt_builder = PromptBuilder()
    llm_client = StubLLMClient()
    qa_pipeline = QAPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm_client,
    )

    app.state.retriever = retriever
    app.state.qa_pipeline = qa_pipeline
    app.state.readiness_service = ReadinessService(
        mode=app.state.api_mode,
        dependencies=app.state.dependencies,
    )
    app.state.search_service = SearchService(
        mode=app.state.api_mode,
        retriever=retriever,
    )
    app.state.chat_service = ChatService(
        mode=app.state.api_mode,
        dependencies=app.state.dependencies,
        qa_pipeline=qa_pipeline,
    )
    app.include_router(system_router)
    app.include_router(search_router)
    app.include_router(chat_router)
    return app


app = create_app()
