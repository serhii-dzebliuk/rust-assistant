from dataclasses import replace
from uuid import uuid4

import pytest

from rust_assistant.application.dto.chunk_context import ChunkContext
from rust_assistant.application.ports.vector_storage import VectorPayload, VectorSearchHit
from rust_assistant.application.use_cases.search import SearchCommand, SearchUseCase
from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.domain.value_objects.identifiers import ChunkId, DocumentId

pytestmark = pytest.mark.unit


class FakeEmbeddingClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.texts = []

    async def embed_text(self, text):
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("embedding failed")
        return [0.1, 0.2, 0.3]


class FakeVectorStorage:
    def __init__(self, *, hits=None, fail: bool = False):
        self.hits = hits or []
        self.fail = fail
        self.calls = []

    async def search(self, query_vector, limit, score_threshold=None):
        self.calls.append(
            {
                "query_vector": query_vector,
                "limit": limit,
                "score_threshold": score_threshold,
            }
        )
        if self.fail:
            raise RuntimeError("vector search failed")
        return self.hits


class FakeChunks:
    def __init__(self, contexts=None, *, fail: bool = False):
        self.contexts = contexts or []
        self.fail = fail
        self.ids = []

    async def get_contexts(self, ids):
        self.ids.append(list(ids))
        if self.fail:
            raise RuntimeError("repository failed")
        return self.contexts


class FakeUnitOfWork:
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


def _context(*, chunk_id=None, document_id=None, text="Async returns a Future."):
    return ChunkContext(
        chunk_id=ChunkId(chunk_id or uuid4()),
        document_id=DocumentId(document_id or uuid4()),
        text=text,
        title="std::keyword::async",
        source_path="std/keyword.async.html",
        url="https://doc.rust-lang.org/std/keyword.async.html",
        section_title="Keyword async",
        section_path=("std::keyword::async", "Keyword async"),
        section_anchor="keyword.async",
        item_path="std::keyword::async",
        crate=Crate.STD,
        item_type=ItemType.KEYWORD,
        rust_version="1.91.1",
        chunk_index=2,
    )


def _hit(chunk_id, *, score):
    return VectorSearchHit(
        chunk_id=chunk_id,
        score=score,
        payload=VectorPayload(document_id=uuid4()),
    )


def _use_case(*, embedding_client=None, vector_storage=None, chunks=None):
    return SearchUseCase(
        embedding_client=embedding_client or FakeEmbeddingClient(),
        vector_storage=vector_storage or FakeVectorStorage(),
        uow=FakeUnitOfWork(chunks or FakeChunks()),
    )


@pytest.mark.asyncio
async def test_search_returns_hydrated_hits_in_vector_order():
    first = _context(text="First chunk explains async functions.")
    second = _context(text="Second chunk explains await points.")
    vector_storage = FakeVectorStorage(
        hits=[
            _hit(second.chunk_id, score=0.91),
            _hit(first.chunk_id, score=0.87),
        ]
    )
    chunks = FakeChunks(contexts=[second, first])

    result = await _use_case(vector_storage=vector_storage, chunks=chunks).execute(
        SearchCommand(query=" async ", limit=2)
    )

    assert result.query == "async"
    assert [hit.chunk_id for hit in result.hits] == [second.chunk_id, first.chunk_id]
    assert result.hits[0].document_id == second.document_id
    assert result.hits[0].title == "std::keyword::async"
    assert result.hits[0].url == "https://doc.rust-lang.org/std/keyword.async.html"
    assert result.hits[0].section == "Keyword async"
    assert result.hits[0].crate == "std"
    assert result.hits[0].item_type == "keyword"
    assert result.hits[0].rust_version == "1.91.1"
    assert result.hits[0].score == 0.91
    assert result.hits[0].text == "Second chunk explains await points."
    assert chunks.ids == [[second.chunk_id, first.chunk_id]]


@pytest.mark.asyncio
async def test_search_calls_vector_storage_with_query_vector_and_limit():
    vector_storage = FakeVectorStorage()

    await _use_case(vector_storage=vector_storage).execute(
        SearchCommand(query="async", limit=5)
    )

    assert vector_storage.calls[0] == {
        "query_vector": [0.1, 0.2, 0.3],
        "limit": 5,
        "score_threshold": None,
    }


@pytest.mark.asyncio
async def test_search_skips_missing_chunk_contexts_without_reordering_found_hits():
    first = _context()
    missing = ChunkId(uuid4())
    second = replace(_context(), chunk_id=ChunkId(uuid4()))
    vector_storage = FakeVectorStorage(
        hits=[
            _hit(first.chunk_id, score=0.95),
            _hit(missing, score=0.93),
            _hit(second.chunk_id, score=0.9),
        ]
    )
    chunks = FakeChunks(contexts=[second, first])

    result = await _use_case(vector_storage=vector_storage, chunks=chunks).execute(
        SearchCommand(query="async", limit=3)
    )

    assert [hit.chunk_id for hit in result.hits] == [first.chunk_id, second.chunk_id]


@pytest.mark.asyncio
async def test_search_returns_empty_result_without_loading_contexts():
    chunks = FakeChunks()

    result = await _use_case(chunks=chunks).execute(SearchCommand(query="async", limit=5))

    assert result.hits == []
    assert chunks.ids == []


@pytest.mark.asyncio
async def test_search_propagates_dependency_failures():
    with pytest.raises(RuntimeError, match="embedding failed"):
        await _use_case(embedding_client=FakeEmbeddingClient(fail=True)).execute(
            SearchCommand(query="async", limit=5)
        )

    with pytest.raises(RuntimeError, match="vector search failed"):
        await _use_case(vector_storage=FakeVectorStorage(fail=True)).execute(
            SearchCommand(query="async", limit=5)
        )

    context = _context()
    with pytest.raises(RuntimeError, match="repository failed"):
        await _use_case(
            vector_storage=FakeVectorStorage(hits=[_hit(context.chunk_id, score=0.8)]),
            chunks=FakeChunks(fail=True),
        ).execute(SearchCommand(query="async", limit=5))
