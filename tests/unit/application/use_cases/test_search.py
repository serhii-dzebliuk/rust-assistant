from dataclasses import replace
from uuid import uuid4

import pytest

from rust_assistant.application.dto.chunk_context import ChunkContext
from rust_assistant.application.ports.reranking_client import RerankingResult
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


class FakeRerankingClient:
    def __init__(self, *, results=None, fail: bool = False):
        self.results = results
        self.fail = fail
        self.calls = []

    async def rerank(self, query, candidates):
        self.calls.append(
            {
                "query": query,
                "candidates": list(candidates),
            }
        )
        if self.fail:
            raise RuntimeError("reranking failed")
        if self.results is not None:
            return self.results
        return [
            RerankingResult(chunk_id=candidate.chunk_id, score=1.0 - index / 10)
            for index, candidate in enumerate(candidates)
        ]


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


def _use_case(
    *,
    embedding_client=None,
    vector_storage=None,
    reranking_client=None,
    chunks=None,
):
    return SearchUseCase(
        embedding_client=embedding_client or FakeEmbeddingClient(),
        vector_storage=vector_storage or FakeVectorStorage(),
        reranking_client=reranking_client or FakeRerankingClient(),
        uow=FakeUnitOfWork(chunks or FakeChunks()),
    )


@pytest.mark.asyncio
async def test_search_returns_hydrated_hits_in_reranker_order():
    first = _context(text="First chunk explains async functions.")
    second = _context(text="Second chunk explains await points.")
    vector_storage = FakeVectorStorage(
        hits=[
            _hit(second.chunk_id, score=0.91),
            _hit(first.chunk_id, score=0.87),
        ]
    )
    chunks = FakeChunks(contexts=[second, first])
    reranking_client = FakeRerankingClient(
        results=[
            RerankingResult(chunk_id=first.chunk_id, score=0.98),
            RerankingResult(chunk_id=second.chunk_id, score=0.84),
        ]
    )

    result = await _use_case(
        vector_storage=vector_storage,
        reranking_client=reranking_client,
        chunks=chunks,
    ).execute(SearchCommand(query=" async ", retrieval_limit=2, reranking_limit=2))

    assert result.query == "async"
    assert [hit.chunk_id for hit in result.hits] == [first.chunk_id, second.chunk_id]
    assert result.hits[0].document_id == first.document_id
    assert result.hits[0].title == "std::keyword::async"
    assert result.hits[0].url == "https://doc.rust-lang.org/std/keyword.async.html"
    assert result.hits[0].section == "Keyword async"
    assert result.hits[0].crate == "std"
    assert result.hits[0].item_type == "keyword"
    assert result.hits[0].rust_version == "1.91.1"
    assert result.hits[0].score == 0.98
    assert result.hits[0].text == "First chunk explains async functions."
    assert chunks.ids == [[second.chunk_id, first.chunk_id]]
    assert [candidate.chunk_id for candidate in reranking_client.calls[0]["candidates"]] == [
        second.chunk_id,
        first.chunk_id,
    ]


@pytest.mark.asyncio
async def test_search_calls_vector_storage_with_query_vector_and_limit():
    vector_storage = FakeVectorStorage()

    await _use_case(vector_storage=vector_storage).execute(
        SearchCommand(query="async", retrieval_limit=5, reranking_limit=2)
    )

    assert vector_storage.calls[0] == {
        "query_vector": [0.1, 0.2, 0.3],
        "limit": 5,
        "score_threshold": None,
    }


@pytest.mark.asyncio
async def test_search_can_return_vector_only_baseline_without_reranking():
    first = _context(text="First chunk.")
    second = replace(_context(), chunk_id=ChunkId(uuid4()), text="Second chunk.")
    vector_storage = FakeVectorStorage(
        hits=[
            _hit(first.chunk_id, score=0.95),
            _hit(second.chunk_id, score=0.93),
        ]
    )
    reranking_client = FakeRerankingClient()

    result = await _use_case(
        vector_storage=vector_storage,
        reranking_client=reranking_client,
        chunks=FakeChunks(contexts=[first, second]),
    ).execute(
        SearchCommand(
            query="async",
            retrieval_limit=2,
            reranking_limit=1,
            use_reranking=False,
        )
    )

    assert [hit.chunk_id for hit in result.hits] == [first.chunk_id]
    assert result.hits[0].score == 0.95
    assert reranking_client.calls == []


@pytest.mark.asyncio
async def test_search_skips_missing_chunk_contexts_before_reranking():
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
    reranking_client = FakeRerankingClient()

    result = await _use_case(
        vector_storage=vector_storage,
        reranking_client=reranking_client,
        chunks=chunks,
    ).execute(SearchCommand(query="async", retrieval_limit=3, reranking_limit=3))

    assert [hit.chunk_id for hit in result.hits] == [first.chunk_id, second.chunk_id]
    assert [candidate.chunk_id for candidate in reranking_client.calls[0]["candidates"]] == [
        first.chunk_id,
        second.chunk_id,
    ]


@pytest.mark.asyncio
async def test_search_truncates_results_to_reranking_limit():
    first = _context()
    second = replace(_context(), chunk_id=ChunkId(uuid4()))
    vector_storage = FakeVectorStorage(
        hits=[
            _hit(first.chunk_id, score=0.95),
            _hit(second.chunk_id, score=0.93),
        ]
    )
    reranking_client = FakeRerankingClient(
        results=[
            RerankingResult(chunk_id=second.chunk_id, score=0.99),
            RerankingResult(chunk_id=first.chunk_id, score=0.88),
        ]
    )

    result = await _use_case(
        vector_storage=vector_storage,
        reranking_client=reranking_client,
        chunks=FakeChunks(contexts=[first, second]),
    ).execute(SearchCommand(query="async", retrieval_limit=2, reranking_limit=1))

    assert [hit.chunk_id for hit in result.hits] == [second.chunk_id]


@pytest.mark.asyncio
async def test_search_returns_empty_result_without_loading_contexts():
    chunks = FakeChunks()
    reranking_client = FakeRerankingClient()

    result = await _use_case(reranking_client=reranking_client, chunks=chunks).execute(
        SearchCommand(query="async", retrieval_limit=5, reranking_limit=2)
    )

    assert result.hits == []
    assert chunks.ids == []
    assert reranking_client.calls == []


@pytest.mark.asyncio
async def test_search_returns_empty_result_without_reranking_when_no_contexts_found():
    context = _context()
    reranking_client = FakeRerankingClient()

    result = await _use_case(
        vector_storage=FakeVectorStorage(hits=[_hit(context.chunk_id, score=0.8)]),
        reranking_client=reranking_client,
        chunks=FakeChunks(contexts=[]),
    ).execute(SearchCommand(query="async", retrieval_limit=5, reranking_limit=2))

    assert result.hits == []
    assert reranking_client.calls == []


@pytest.mark.asyncio
async def test_search_propagates_dependency_failures():
    with pytest.raises(RuntimeError, match="embedding failed"):
        await _use_case(embedding_client=FakeEmbeddingClient(fail=True)).execute(
            SearchCommand(query="async", retrieval_limit=5, reranking_limit=2)
        )

    with pytest.raises(RuntimeError, match="vector search failed"):
        await _use_case(vector_storage=FakeVectorStorage(fail=True)).execute(
            SearchCommand(query="async", retrieval_limit=5, reranking_limit=2)
        )

    context = _context()
    with pytest.raises(RuntimeError, match="repository failed"):
        await _use_case(
            vector_storage=FakeVectorStorage(hits=[_hit(context.chunk_id, score=0.8)]),
            chunks=FakeChunks(fail=True),
        ).execute(SearchCommand(query="async", retrieval_limit=5, reranking_limit=2))

    with pytest.raises(RuntimeError, match="reranking failed"):
        await _use_case(
            vector_storage=FakeVectorStorage(hits=[_hit(context.chunk_id, score=0.8)]),
            reranking_client=FakeRerankingClient(fail=True),
            chunks=FakeChunks(contexts=[context]),
        ).execute(SearchCommand(query="async", retrieval_limit=5, reranking_limit=2))
