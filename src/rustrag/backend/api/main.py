"""FastAPI application entry point for the serving layer."""

from __future__ import annotations

from fastapi import FastAPI

from rustrag.backend.rag import PromptBuilder, QAPipeline, StubLLMClient, StubRetriever
from rustrag.backend.services import ChatService, ReadinessService, SearchService

from .routers import chat_router, search_router, system_router


def create_app() -> FastAPI:
    """Create the FastAPI application for the serving API."""
    app = FastAPI(
        title="RustRAG API",
        version="0.1.0",
    )
    # TODO: Replace the hard-coded stub mode with a real runtime mode loaded
    # from application settings/environment.
    app.state.api_mode = "stub"
    # TODO: Replace placeholder dependency statuses with real clients/health
    # checks once Postgres and Qdrant are configured.
    app.state.dependencies = {
        "postgres": "not_configured",
        "qdrant": "not_configured",
    }

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
    app.include_router(system_router, prefix="/api")
    app.include_router(search_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    return app


app = create_app()
