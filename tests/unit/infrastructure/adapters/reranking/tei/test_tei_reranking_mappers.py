from uuid import uuid4

import pytest

from rust_assistant.application.ports.reranking_client import RerankingCandidate
from rust_assistant.domain.value_objects.identifiers import ChunkId
from rust_assistant.infrastructure.adapters.reranking.tei.mappers import (
    map_reranking_request,
    map_reranking_response,
)

pytestmark = pytest.mark.unit


def _candidate(text: str = "Async returns a Future.") -> RerankingCandidate:
    return RerankingCandidate(chunk_id=ChunkId(uuid4()), text=text)


def test_map_reranking_request_uses_tei_shape():
    first = _candidate("first")
    second = _candidate("second")

    payload = map_reranking_request("async", [first, second])

    assert payload == {
        "query": "async",
        "raw_scores": False,
        "return_text": False,
        "texts": ["first", "second"],
        "truncate": False,
        "truncation_direction": "right",
    }


def test_map_reranking_response_maps_indexes_to_candidate_ids_in_response_order():
    first = _candidate("first")
    second = _candidate("second")

    results = map_reranking_response(
        [
            {"index": 1, "score": 0.98, "text": "second"},
            {"index": 0, "score": 0.87, "text": "first"},
        ],
        [first, second],
    )

    assert [result.chunk_id for result in results] == [second.chunk_id, first.chunk_id]
    assert [result.score for result in results] == [0.98, 0.87]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"index": 0, "score": 0.9}, "must be a list"),
        ([None], "must be an object"),
        ([{"index": "0", "score": 0.9}], "index must be an integer"),
        ([{"index": True, "score": 0.9}], "index must be an integer"),
        ([{"index": 2, "score": 0.9}], "index out of range"),
        ([{"index": 0, "score": "0.9"}], "score must be a number"),
        ([{"index": 0, "score": False}], "score must be a number"),
        ([{"index": 0, "score": 0.9}, {"index": 0, "score": 0.8}], "duplicate index"),
    ],
)
def test_map_reranking_response_rejects_invalid_payloads(payload, match):
    with pytest.raises(ValueError, match=match):
        map_reranking_response(payload, [_candidate(), _candidate()])
