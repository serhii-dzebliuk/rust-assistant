from uuid import UUID

import pytest

from rust_assistant.application.ports.vector_storage import VectorPayload
from rust_assistant.infrastructure.adapters.vector_storage.qdrant.mappers import (
    map_filters_to_qdrant_filter,
    map_vector_payload_from_qdrant_payload,
    map_vector_payload_to_qdrant_payload,
)

pytestmark = pytest.mark.unit


DOCUMENT_ID = UUID("1a0b5f1e-b466-5c53-858f-7d6d50c8d8c8")


def test_map_vector_payload_to_qdrant_payload_omits_none_values():
    payload = VectorPayload(
        document_id=DOCUMENT_ID,
        crate="std",
        item_type=None,
        item_path="std::primitive::unit",
        section_title=None,
        chunk_index=0,
    )

    qdrant_payload = map_vector_payload_to_qdrant_payload(payload)

    assert qdrant_payload == {
        "document_id": "1a0b5f1e-b466-5c53-858f-7d6d50c8d8c8",
        "crate": "std",
        "item_path": "std::primitive::unit",
        "chunk_index": 0,
    }
    assert "item_type" not in qdrant_payload
    assert "section_title" not in qdrant_payload


def test_map_vector_payload_from_qdrant_payload_restores_vector_payload():
    payload = map_vector_payload_from_qdrant_payload(
        {
            "document_id": str(DOCUMENT_ID),
            "crate": "std",
            "item_type": "primitive",
            "item_path": "std::primitive::unit",
            "section_title": "Primitive Type unit",
            "chunk_index": 2,
        }
    )

    assert payload == VectorPayload(
        document_id=DOCUMENT_ID,
        crate="std",
        item_type="primitive",
        item_path="std::primitive::unit",
        section_title="Primitive Type unit",
        chunk_index=2,
    )


def test_map_vector_payload_from_qdrant_payload_requires_document_id():
    with pytest.raises(ValueError, match="document_id"):
        map_vector_payload_from_qdrant_payload({"crate": "std"})


def test_map_filters_to_qdrant_filter_maps_supported_equality_conditions():
    qdrant_filter = map_filters_to_qdrant_filter(
        {
            "document_id": DOCUMENT_ID,
            "crate": "std",
            "chunk_index": 3,
        }
    )

    assert qdrant_filter is not None
    assert qdrant_filter.must is not None
    conditions = {condition.key: condition.match.value for condition in qdrant_filter.must}
    assert conditions == {
        "document_id": "1a0b5f1e-b466-5c53-858f-7d6d50c8d8c8",
        "crate": "std",
        "chunk_index": 3,
    }


def test_map_filters_to_qdrant_filter_returns_none_for_empty_filters():
    assert map_filters_to_qdrant_filter(None) is None
    assert map_filters_to_qdrant_filter({}) is None


def test_map_filters_to_qdrant_filter_rejects_unknown_filter_key():
    with pytest.raises(ValueError, match="Unsupported Qdrant filter field"):
        map_filters_to_qdrant_filter({"source_path": "std/primitive.unit.html"})


def test_map_filters_to_qdrant_filter_rejects_unsupported_filter_value_shape():
    with pytest.raises(ValueError, match="Unsupported Qdrant filter value"):
        map_filters_to_qdrant_filter({"crate": ["std", "core"]})
