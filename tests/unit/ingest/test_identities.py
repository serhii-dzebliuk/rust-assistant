from dataclasses import replace

import pytest

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate

pytestmark = pytest.mark.unit


def _document(source_path: str = "std/keyword.async.html") -> Document:
    return Document(
        source_path=source_path,
        title="std::keyword::async",
        text="Keyword async",
        crate=Crate.STD,
        url=f"https://doc.rust-lang.org/{source_path}",
    )


def _chunk(source_path: str = "std/keyword.async.html", chunk_index: int = 0) -> Chunk:
    return Chunk(
        source_path=source_path,
        chunk_index=chunk_index,
        text="Returns a Future.",
        crate=Crate.STD,
        start_offset=0,
        end_offset=17,
    )


def test_document_id_is_stable_for_the_same_source_path():
    first = _document()
    second = _document()

    assert first.id == second.id


def test_document_id_changes_when_source_path_changes():
    first = _document("std/keyword.async.html")
    second = _document("std/keyword.await.html")

    assert first.id != second.id


def test_chunk_identity_is_stable_for_the_same_document_and_index():
    first = _chunk()
    second = _chunk()

    assert first.document_id == second.document_id
    assert first.id == second.id


def test_chunk_identity_changes_when_chunk_index_changes():
    first = _chunk(chunk_index=0)
    second = replace(first, chunk_index=1)

    assert first.id != second.id
