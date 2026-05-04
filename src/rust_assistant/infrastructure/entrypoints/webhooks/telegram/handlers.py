"""aiogram handlers for the Telegram webhook frontend."""

from __future__ import annotations

import logging
from typing import Optional

from aiogram import F, Dispatcher, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from rust_assistant.application.use_cases.chat import (
    ChatCommand,
    ChatQuestionTooLargeError,
    ChatUseCase,
)

logger = logging.getLogger(__name__)

MAX_TELEGRAM_MESSAGE_CHARS = 4000

router = Router(name="telegram_webhook")


def register_telegram_handlers(dispatcher: Dispatcher) -> None:
    """Attach Telegram message handlers to the aiogram dispatcher."""
    dispatcher.include_router(router)


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Send a short bot introduction."""
    await message.answer(
        "Hi. I am a Rust documentation RAG assistant.\n\n"
        "Send me a Rust question, and I will answer using the indexed docs."
    )


@router.message(F.text)
async def handle_text_message(
    message: Message,
    chat_use_case: Optional[ChatUseCase],
) -> None:
    """Answer one text question through the chat use case."""
    question = message.text.strip() if message.text is not None else ""
    if not question:
        await message.answer("Please send a non-empty text question.")
        return

    if chat_use_case is None:
        await message.answer("Chat is not configured yet. Please try again later.")
        return

    try:
        result = await chat_use_case.execute(ChatCommand(question=question))
    except ChatQuestionTooLargeError as exc:
        await message.answer(str(exc))
        return
    except Exception:
        logger.exception("Telegram chat answer generation failed")
        await message.answer("Sorry, I could not generate an answer right now.")
        return

    for part in _split_telegram_text(result.answer):
        await message.answer(part)


@router.message()
async def handle_non_text_message(message: Message) -> None:
    """Reject unsupported Telegram message types."""
    await message.answer("Please send a text question.")


def _split_telegram_text(
    text: str,
    *,
    max_length: int = MAX_TELEGRAM_MESSAGE_CHARS,
) -> list[str]:
    """Split long answers into Telegram-sized text messages."""
    normalized = text.strip()
    if not normalized:
        return ["I do not have an answer for that yet."]

    parts: list[str] = []
    remaining = normalized
    while len(remaining) > max_length:
        split_at = max(
            remaining.rfind("\n", 0, max_length),
            remaining.rfind(" ", 0, max_length),
        )
        if split_at < max_length // 2:
            split_at = max_length
        part = remaining[:split_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts
