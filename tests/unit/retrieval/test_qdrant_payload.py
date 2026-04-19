import pytest

from rust_assistant.models import ChunkRecord, DocumentRecord
from rust_assistant.retrieval.qdrant_payload import build_chunk_payload

pytestmark = pytest.mark.unit


def test_build_chunk_payload_uses_minimal_metadata_without_text():
    document = DocumentRecord(
        source_path="std/primitive.unit.html",
        crate="std",
        title="std::primitive::unit",
        text_content="Primitive Type unit",
        parsed_content=[],
        url="https://doc.rust-lang.org/std/primitive.unit.html",
        item_path="std::primitive::unit",
        item_type="primitive",
        rust_version="1.91.1",
    )
    document.id = 110
    chunk = ChunkRecord(
        document_id=110,
        text="The `()` type, also called unit.",
        hash="abc123",
        chunk_index=0,
    )
    chunk.id = 4695
    chunk.document = document

    payload = build_chunk_payload(chunk)

    assert payload == {
        "chunk_id": 4695,
        "document_id": 110,
        "crate": "std",
        "item_type": "primitive",
        "rust_version": "1.91.1",
        "source_path": "std/primitive.unit.html",
        "item_path": "std::primitive::unit",
        "chunk_index": 0,
        "hash": "abc123",
    }
    assert "text" not in payload
