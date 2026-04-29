"""FastAPI application wiring for the ASGI entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from rust_assistant.bootstrap.container import RuntimeContainer, build_container
from rust_assistant.infrastructure.entrypoints.api.routers.chat import router as chat_router
from rust_assistant.infrastructure.entrypoints.api.routers.search import router as search_router
from rust_assistant.infrastructure.entrypoints.api.routers.system import router as system_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close runtime resources owned by the app container on shutdown."""
    try:
        yield
    finally:
        close = getattr(app.state.container, "aclose", None)
        if close is not None:
            await close()


def create_app(*, container: Optional[RuntimeContainer] = None) -> FastAPI:
    """Create the FastAPI application for the serving API."""
    runtime_container = container or build_container(include_search=True)
    logger.info("Creating FastAPI application")

    app = FastAPI(
        title="Rust Assistant API",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.container = runtime_container
    app.include_router(system_router)
    app.include_router(search_router)
    app.include_router(chat_router)
    return app
