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
from rust_assistant.infrastructure.entrypoints.api.routers.webhooks import router as webhooks_router
from rust_assistant.infrastructure.entrypoints.webhooks.telegram.constants import (
    TELEGRAM_ALLOWED_UPDATES,
    TELEGRAM_DROP_PENDING_UPDATES,
    TELEGRAM_WEBHOOK_PATH,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close runtime resources owned by the app container on shutdown."""
    try:
        await _configure_telegram_webhook(app)
        yield
    finally:
        close = getattr(app.state.container, "aclose", None)
        if close is not None:
            await close()


async def _configure_telegram_webhook(app: FastAPI) -> None:
    """Register Telegram webhook settings when Telegram runtime is configured."""
    container = app.state.container
    bot = getattr(container, "telegram_bot", None)
    if bot is None:
        return

    settings = container.settings
    webhook_secret = settings.telegram.webhook_secret
    public_base_url = settings.proxy.public_base_url
    if public_base_url is None or webhook_secret is None:
        logger.warning("Telegram bot is configured, but webhook registration settings are missing")
        return

    try:
        from aiogram.types import BotCommand
    except ImportError:
        logger.exception("aiogram must be installed before registering Telegram webhook")
        raise

    webhook_url = f"{public_base_url.rstrip('/')}{TELEGRAM_WEBHOOK_PATH}"
    await bot.set_webhook(
        webhook_url,
        allowed_updates=TELEGRAM_ALLOWED_UPDATES,
        drop_pending_updates=TELEGRAM_DROP_PENDING_UPDATES,
        secret_token=webhook_secret,
    )
    await bot.set_my_commands(
        [
            BotCommand(
                command="start",
                description="About this Rust RAG assistant",
            )
        ]
    )
    logger.info("Telegram webhook registered: url=%s", webhook_url)


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
    app.include_router(webhooks_router)
    return app
