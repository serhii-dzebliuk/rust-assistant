from dataclasses import replace

import pytest

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.infrastructure.outbound.tokenizers.transformers_chunk_counter import (
    TransformersChunkTokenCounter,
)

pytestmark = pytest.mark.unit


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        return list(range(len(text.split())))


def _chunk(text: str = "Returns a Future instead of blocking.") -> Chunk:
    return Chunk(
        source_path="std/keyword.async.html",
        chunk_index=0,
        text=text,
        crate=Crate.STD,
        start_offset=0,
        end_offset=len(text),
        item_path="std::keyword::async",
        item_type=ItemType.UNKNOWN,
        rust_version="1.91.1",
        url="https://doc.rust-lang.org/std/keyword.async.html",
        section_path=["std::keyword::async"],
        section_anchor="keyword.async",
    )


def test_chunk_token_counter_copies_chunks_with_model_token_counts():
    tokenizer = FakeTokenizer()
    counter = TransformersChunkTokenCounter(
        model_name="fake-embedding-model",
        tokenizer=tokenizer,
    )
    chunk = _chunk("Returns a Future.")

    counted_chunks = counter.with_token_counts([chunk])

    assert chunk.token_count is None
    assert counted_chunks == [replace(chunk, token_count=3)]
    assert tokenizer.calls == [("Returns a Future.", False)]


def test_chunk_token_counter_rejects_blank_model_name():
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        TransformersChunkTokenCounter.from_model_name(" ")
