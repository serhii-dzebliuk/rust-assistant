from uuid import UUID

import pytest
from qdrant_client.http import models as qdrant_models

from rust_assistant.application.ports.vector_storage import VectorPayload, VectorPoint
from rust_assistant.infrastructure.adapters.vector_storage.qdrant.qdrant_vector_storage import QdrantVectorStorage

pytestmark = pytest.mark.unit


CHUNK_ID = UUID("4c1b52fd-12dc-5565-92b0-1b52b67bf809")
DOCUMENT_ID = UUID("1a0b5f1e-b466-5c53-858f-7d6d50c8d8c8")


class FakeQdrantClient:
    def __init__(self):
        self.recreate_collection_calls = []
        self.upsert_calls = []
        self.query_points_calls = []
        self.query_points_result = qdrant_models.QueryResponse(points=[])

    async def recreate_collection(self, **kwargs):
        self.recreate_collection_calls.append(kwargs)
        return True

    async def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)
        return None

    async def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return self.query_points_result


def _storage(client):
    return QdrantVectorStorage(
        client=client,
        collection_name="rust-docs",
        vector_size=3,
        distance="cosine",
    )


def _storage_with_batch_size(client, batch_size):
    return QdrantVectorStorage(
        client=client,
        collection_name="rust-docs",
        vector_size=3,
        distance="cosine",
        upsert_batch_size=batch_size,
    )


def _vector_payload():
    return VectorPayload(
        document_id=DOCUMENT_ID,
        crate="std",
        item_type="primitive",
        source_path="std/primitive.unit.html",
        item_path="std::primitive::unit",
        rust_version="1.91.1",
        section_title="Primitive Type unit",
        chunk_index=0,
        text_hash="abc123",
    )


def _vector_point(index):
    return VectorPoint(
        chunk_id=UUID(f"00000000-0000-4000-8000-{index:012d}"),
        vector=[0.1, 0.2, 0.3],
        payload=_vector_payload(),
    )


@pytest.mark.asyncio
async def test_recreate_collection_passes_collection_name_vector_size_and_distance():
    client = FakeQdrantClient()

    await _storage(client).recreate_collection()

    assert len(client.recreate_collection_calls) == 1
    call = client.recreate_collection_calls[0]
    assert call["collection_name"] == "rust-docs"
    assert call["vectors_config"].size == 3
    assert call["vectors_config"].distance == qdrant_models.Distance.COSINE


def test_constructor_accepts_distance_case_insensitively():
    QdrantVectorStorage(
        client=FakeQdrantClient(),
        collection_name="rust-docs",
        vector_size=3,
        distance="COSINE",
    )
    QdrantVectorStorage(
        client=FakeQdrantClient(),
        collection_name="rust-docs",
        vector_size=3,
        distance="Cosine",
    )


def test_constructor_rejects_unknown_distance():
    with pytest.raises(ValueError, match="Unsupported Qdrant distance"):
        QdrantVectorStorage(
            client=FakeQdrantClient(),
            collection_name="rust-docs",
            vector_size=3,
            distance="chebyshev",
        )


def test_constructor_rejects_invalid_upsert_batch_size():
    with pytest.raises(ValueError, match="upsert_batch_size"):
        _storage_with_batch_size(FakeQdrantClient(), 0)


@pytest.mark.asyncio
async def test_upsert_vectors_sends_point_structs_with_serialized_payload():
    client = FakeQdrantClient()
    point = VectorPoint(
        chunk_id=CHUNK_ID,
        vector=[0.1, 0.2, 0.3],
        payload=_vector_payload(),
    )

    await _storage(client).upsert_vectors([point])

    assert len(client.upsert_calls) == 1
    call = client.upsert_calls[0]
    assert call["collection_name"] == "rust-docs"
    assert len(call["points"]) == 1
    qdrant_point = call["points"][0]
    assert qdrant_point.id == CHUNK_ID
    assert qdrant_point.vector == [0.1, 0.2, 0.3]
    assert qdrant_point.payload == {
        "document_id": "1a0b5f1e-b466-5c53-858f-7d6d50c8d8c8",
        "crate": "std",
        "item_type": "primitive",
        "source_path": "std/primitive.unit.html",
        "item_path": "std::primitive::unit",
        "rust_version": "1.91.1",
        "section_title": "Primitive Type unit",
        "chunk_index": 0,
        "text_hash": "abc123",
    }


@pytest.mark.asyncio
async def test_upsert_vectors_is_noop_for_empty_points():
    client = FakeQdrantClient()

    await _storage(client).upsert_vectors([])

    assert client.upsert_calls == []


@pytest.mark.asyncio
async def test_upsert_vectors_splits_large_inputs_into_batches():
    client = FakeQdrantClient()

    await _storage_with_batch_size(client, 2).upsert_vectors(
        [_vector_point(1), _vector_point(2), _vector_point(3), _vector_point(4), _vector_point(5)]
    )

    assert len(client.upsert_calls) == 3
    assert [len(call["points"]) for call in client.upsert_calls] == [2, 2, 1]
    assert [call["collection_name"] for call in client.upsert_calls] == [
        "rust-docs",
        "rust-docs",
        "rust-docs",
    ]


@pytest.mark.asyncio
async def test_search_calls_query_points_with_filters_and_maps_hits():
    client = FakeQdrantClient()
    client.query_points_result = qdrant_models.QueryResponse(
        points=[
            qdrant_models.ScoredPoint(
                id=CHUNK_ID,
                version=1,
                score=0.92,
                payload={
                    "document_id": str(DOCUMENT_ID),
                    "crate": "std",
                    "item_type": "primitive",
                    "source_path": "std/primitive.unit.html",
                    "item_path": "std::primitive::unit",
                    "rust_version": "1.91.1",
                    "section_title": "Primitive Type unit",
                    "chunk_index": 0,
                    "text_hash": "abc123",
                },
            )
        ]
    )

    hits = await _storage(client).search(
        query_vector=[0.1, 0.2, 0.3],
        limit=5,
        score_threshold=0.7,
        filters={"crate": "std"},
    )

    assert len(client.query_points_calls) == 1
    call = client.query_points_calls[0]
    assert call["collection_name"] == "rust-docs"
    assert call["query"] == [0.1, 0.2, 0.3]
    assert call["limit"] == 5
    assert call["score_threshold"] == 0.7
    assert call["with_payload"] is True
    assert call["with_vectors"] is False
    assert call["query_filter"].must[0].key == "crate"
    assert call["query_filter"].must[0].match.value == "std"
    assert hits[0].chunk_id == CHUNK_ID
    assert hits[0].score == 0.92
    assert hits[0].payload == _vector_payload()
