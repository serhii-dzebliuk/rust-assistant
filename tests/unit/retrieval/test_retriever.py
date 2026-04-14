import asyncio

import pytest

from rust_assistant.clients.vectordb import VectorSearchHit
from rust_assistant.models import ChunkRecord, DocumentRecord
from rust_assistant.retrieval.retriever import DatabaseBackedRetriever

pytestmark = pytest.mark.unit


class FakeVectorStore:
    async def search(self, *, query, k, filters=None):
        return [VectorSearchHit(chunk_id=42, score=0.91)]


class FakeChunkRepository:
    async def list_by_chunk_ids(self, _session, chunk_ids):
        assert chunk_ids == [42]
        document = DocumentRecord(
            source_path="std/keyword.async.html",
            crate="std",
            title="std::keyword::async",
            text_content="Keyword async",
            parsed_content=[],
            url="https://doc.rust-lang.org/std/keyword.async.html",
            item_path="std::keyword::async",
            item_type="unknown",
        )
        document.id = 7
        chunk = ChunkRecord(
            document_id=7,
            text="Returns a Future instead of blocking the current thread.",
            hash="hash",
            chunk_index=0,
            section_title="Keyword async",
        )
        chunk.id = 42
        chunk.document = document
        return [chunk]


def test_database_backed_retriever_hydrates_numeric_chunk_ids_from_documents():
    retriever = DatabaseBackedRetriever(
        vector_store=FakeVectorStore(),
        chunk_repository=FakeChunkRepository(),
    )

    result = asyncio.run(retriever.search(query="async", k=5, session=object()))

    assert result.total_results == 1
    hit = result.hits[0]
    assert hit.title == "std::keyword::async"
    assert hit.source_path == "std/keyword.async.html"
    assert hit.section == "Keyword async"
    assert hit.item_path == "std::keyword::async"
    assert hit.crate == "std"
    assert hit.item_type == "unknown"
    assert hit.metadata == {"chunk_id": 42}
