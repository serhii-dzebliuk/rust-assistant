"""Inbound webhook endpoints."""

from __future__ import annotations

import json
import logging
import secrets
from typing import Any

from aiogram.types import Update
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import ValidationError

from rust_assistant.infrastructure.entrypoints.webhooks.telegram.constants import (
    TELEGRAM_WEBHOOK_PATH,
)

logger = logging.getLogger(__name__)

TELEGRAM_SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"

router = APIRouter(tags=["webhooks"])


@router.post(TELEGRAM_WEBHOOK_PATH)
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, bool]:
    """Accept one Telegram update and dispatch it to aiogram in the background."""
    container = request.app.state.container
    bot = getattr(container, "telegram_bot", None)
    dispatcher = getattr(container, "telegram_dispatcher", None)

    if bot is None or dispatcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram webhook is not configured",
        )

    expected_secret = getattr(container.settings.telegram, "webhook_secret", None)
    if expected_secret is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram webhook is not configured",
        )

    provided_secret = request.headers.get(TELEGRAM_SECRET_HEADER)
    if provided_secret is None or not secrets.compare_digest(provided_secret, expected_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Telegram webhook secret",
        )

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed Telegram webhook payload",
        ) from exc

    update = _parse_update(payload, bot)
    background_tasks.add_task(dispatcher.feed_update, bot, update)
    return {"ok": True}


def _parse_update(payload: Any, bot: Any) -> Update:
    """Parse raw Telegram webhook JSON into an aiogram Update."""
    try:
        return Update.model_validate(payload, context={"bot": bot})
    except ValidationError as exc:
        logger.warning("Malformed Telegram update received: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Malformed Telegram update",
        ) from exc
