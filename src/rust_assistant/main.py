"""FastAPI application entry point for the serving layer."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from rust_assistant.api.routers import chat_router, search_router, system_router
from rust_assistant.core.config import get_settings
from rust_assistant.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the FastAPI application for the serving API."""

    settings = get_settings()
    configure_logging(logging_settings=settings.logging)
    logger.info("Creating FastAPI application")

    app = FastAPI(
        title="Rust Assistant API",
        version="0.1.0",
    )

    app.include_router(system_router)
    app.include_router(search_router)
    app.include_router(chat_router)
    return app


app = create_app()
