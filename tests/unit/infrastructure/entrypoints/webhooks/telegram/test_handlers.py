from datetime import datetime, timezone

import pytest
from aiogram.types import Chat, Message

from rust_assistant.application.use_cases.chat import ChatQuestionTooLargeError, ChatResult
from rust_assistant.infrastructure.entrypoints.webhooks.telegram.handlers import (
    MAX_TELEGRAM_MESSAGE_CHARS,
    handle_non_text_message,
    handle_start,
    handle_text_message,
)

pytestmark = pytest.mark.unit


class FakeChatUseCase:
    def __init__(self, answer: str = "Rust answer"):
        self.answer = answer
        self.commands = []
        self.error = None

    async def execute(self, command):
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return ChatResult(answer=self.answer)


@pytest.fixture
def answers(monkeypatch):
    sent = []

    async def fake_answer(self, text, *args, **kwargs):
        sent.append(text)

    monkeypatch.setattr(Message, "answer", fake_answer)
    return sent


def _message(*, text=None) -> Message:
    return Message(
        message_id=1,
        date=datetime.fromtimestamp(1, tz=timezone.utc),
        chat=Chat(id=10, type="private"),
        text=text,
    )


@pytest.mark.asyncio
async def test_start_handler_sends_intro_text(answers):
    await handle_start(_message(text="/start"))

    assert len(answers) == 1
    assert "Rust documentation RAG assistant" in answers[0]


@pytest.mark.asyncio
async def test_text_handler_maps_message_to_chat_use_case(answers):
    use_case = FakeChatUseCase(answer="Async answer")

    await handle_text_message(_message(text=" What is async? "), use_case)

    assert answers == ["Async answer"]
    assert use_case.commands[0].question == "What is async?"


@pytest.mark.asyncio
async def test_text_handler_splits_long_answers(answers):
    use_case = FakeChatUseCase(answer="a" * (MAX_TELEGRAM_MESSAGE_CHARS * 2 + 1))

    await handle_text_message(_message(text="long answer please"), use_case)

    assert len(answers) == 3
    assert all(len(answer) <= MAX_TELEGRAM_MESSAGE_CHARS for answer in answers)


@pytest.mark.asyncio
async def test_text_handler_returns_validation_message_for_large_question(answers):
    use_case = FakeChatUseCase()
    use_case.error = ChatQuestionTooLargeError("Question is too large")

    await handle_text_message(_message(text="oversized"), use_case)

    assert answers == ["Question is too large"]


@pytest.mark.asyncio
async def test_non_text_handler_does_not_call_chat_use_case(answers):
    await handle_non_text_message(_message())

    assert answers == ["Please send a text question."]
