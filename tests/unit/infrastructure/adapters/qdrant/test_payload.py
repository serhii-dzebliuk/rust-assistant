from uuid import UUID

import pytest

from rust_assistant.infrastructure.adapters.qdrant.payload import build_chunk_payload
from rust_assistant.infrastructure.adapters.sqlalchemy.models import (
    ChunkRecord,
    DocumentRecord,
)

pytestmark = pytest.mark.unit


def test_build_chunk_payload_uses_minimal_metadata_without_text():
    document = DocumentRecord(
        pk=1,
        id=UUID("1a0b5f1e-b466-5c53-858f-7d6d50c8d8c8"),
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
    chunk = ChunkRecord(
        pk=2,
        id=UUID("4c1b52fd-12dc-5565-92b0-1b52b67bf809"),
        document_pk=1,
        text="The `()` type, also called unit.",
        hash="abc123",
        chunk_index=0,
    )
    chunk.document = document

    payload = build_chunk_payload(chunk)

    assert payload == {
        "chunk_id": "4c1b52fd-12dc-5565-92b0-1b52b67bf809",
        "document_id": "1a0b5f1e-b466-5c53-858f-7d6d50c8d8c8",
        "crate": "std",
        "item_type": "primitive",
        "rust_version": "1.91.1",
        "source_path": "std/primitive.unit.html",
        "item_path": "std::primitive::unit",
        "chunk_index": 0,
        "hash": "abc123",
    }
    assert "text" not in payload
