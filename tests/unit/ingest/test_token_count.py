import pytest

from rust_assistant.ingest.entities import Chunk, ChunkMetadata
from rust_assistant.ingest.token_count import ChunkTokenCounter
from rust_assistant.schemas.enums import Crate, ItemType

pytestmark = pytest.mark.unit


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append((text, add_special_tokens))
        return list(range(len(text.split())))


def _chunk(text: str = "Returns a Future instead of blocking.") -> Chunk:
    return Chunk(
        chunk_id="transient-chunk-id",
        doc_id="transient-doc-id",
        text=text,
        metadata=ChunkMetadata(
            crate=Crate.STD,
            item_path="std::keyword::async",
            item_type=ItemType.UNKNOWN,
            rust_version="1.91.1",
            url="https://doc.rust-lang.org/std/keyword.async.html",
            section="Keyword async",
            section_path=["std::keyword::async"],
            anchor="keyword.async",
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            doc_title="std::keyword::async",
            doc_source_path="std/keyword.async.html",
        ),
    )


def test_chunk_token_counter_copies_chunks_with_model_token_counts():
    tokenizer = FakeTokenizer()
    counter = ChunkTokenCounter(model_name="fake-embedding-model", tokenizer=tokenizer)
    chunk = _chunk("Returns a Future.")

    counted_chunks = counter.with_token_counts([chunk])

    assert chunk.token_count is None
    assert counted_chunks[0].token_count == 3
    assert counted_chunks[0].text == chunk.text
    assert counted_chunks[0].metadata == chunk.metadata
    assert tokenizer.calls == [("Returns a Future.", False)]


def test_chunk_token_counter_rejects_blank_model_name():
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        ChunkTokenCounter.from_model_name(" ")
