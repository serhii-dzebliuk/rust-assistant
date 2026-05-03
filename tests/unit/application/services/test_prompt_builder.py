from uuid import UUID

import pytest

from rust_assistant.application.services.prompt_builder import PromptBuilder
from rust_assistant.application.services.retrieval.models import RetrievedChunk

pytestmark = pytest.mark.unit


class FakeTokenizer:
    def __init__(self, counts):
        self.counts = counts

    def count_tokens(self, text):
        return self.counts[text]


def _chunk(
    *,
    text,
    title="std::vec::Vec",
    item_path="std::vec::Vec::push",
    crate="std",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID("11111111-1111-4111-8111-111111111111"),
        document_id=UUID("22222222-2222-4222-8222-222222222222"),
        title=title,
        source_path="std/vec/struct.Vec.html",
        url="https://doc.rust-lang.org/std/vec/struct.Vec.html",
        section="push",
        item_path=item_path,
        crate=crate,
        item_type="method",
        rust_version="1.91.1",
        score=0.99,
        text=text,
    )


def test_prompt_builder_returns_separate_system_and_user_prompts():
    chunk = _chunk(text="The push method appends an element.")
    tokenizer = FakeTokenizer({chunk.text: 6})

    prompt = PromptBuilder(tokenizer=tokenizer, max_context_tokens=10).build(
        question="How does Vec::push work?",
        chunks=[chunk],
    )

    assert "technical assistant for the Rust programming language" in prompt.system_prompt
    assert prompt.user_prompt == (
        "Question:\n"
        "How does Vec::push work?\n\n"
        "Context:\n"
        "[1] std::vec::Vec::push | std | token_count=6\n"
        "The push method appends an element."
    )
    assert prompt.context_chunks == [chunk]


def test_prompt_builder_keeps_first_chunks_that_fit_context_budget():
    first = _chunk(text="first")
    second = _chunk(text="second")
    third = _chunk(text="third")
    tokenizer = FakeTokenizer({"first": 4, "second": 7, "third": 5})

    prompt = PromptBuilder(tokenizer=tokenizer, max_context_tokens=10).build(
        question="question",
        chunks=[first, second, third],
    )

    assert prompt.context_chunks == [first]
    assert "[1] std::vec::Vec::push | std | token_count=4\nfirst" in prompt.user_prompt
    assert "second" not in prompt.user_prompt
    assert "third" not in prompt.user_prompt
