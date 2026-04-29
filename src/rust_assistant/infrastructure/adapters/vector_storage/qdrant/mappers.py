"""Qdrant mappers for vector-storage DTOs."""

from __future__ import annotations

from typing import Any, Final, Optional, Union
from uuid import UUID

from qdrant_client.http import models as qdrant_models

from rust_assistant.application.ports.vector_storage import VectorPayload

_PAYLOAD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "document_id",
        "crate",
        "item_type",
        "source_path",
        "item_path",
        "rust_version",
        "section_title",
        "chunk_index",
        "text_hash",
    }
)


def map_vector_payload_to_qdrant_payload(payload: VectorPayload) -> dict[str, Any]:
    """Convert an application vector payload into a Qdrant payload dict."""
    raw_payload: dict[str, Any] = {
        "document_id": str(payload.document_id),
        "crate": payload.crate,
        "item_type": payload.item_type,
        "source_path": payload.source_path,
        "item_path": payload.item_path,
        "rust_version": payload.rust_version,
        "section_title": payload.section_title,
        "chunk_index": payload.chunk_index,
        "text_hash": payload.text_hash,
    }
    return {key: value for key, value in raw_payload.items() if value is not None}


def map_vector_payload_from_qdrant_payload(payload: dict[str, Any]) -> VectorPayload:
    """Convert a Qdrant payload dict into an application vector payload."""
    document_id = payload.get("document_id")
    if document_id is None:
        raise ValueError("Qdrant payload is missing required document_id")

    try:
        parsed_document_id = UUID(str(document_id))
    except ValueError as exc:
        raise ValueError("Qdrant payload document_id must be a valid UUID") from exc

    return VectorPayload(
        document_id=parsed_document_id,
        crate=_optional_str(payload.get("crate"), "crate"),
        item_type=_optional_str(payload.get("item_type"), "item_type"),
        source_path=_optional_str(payload.get("source_path"), "source_path"),
        item_path=_optional_str(payload.get("item_path"), "item_path"),
        rust_version=_optional_str(payload.get("rust_version"), "rust_version"),
        section_title=_optional_str(payload.get("section_title"), "section_title"),
        chunk_index=_optional_int(payload.get("chunk_index"), "chunk_index"),
        text_hash=_optional_str(payload.get("text_hash"), "text_hash"),
    )


def map_filters_to_qdrant_filter(
    filters: Optional[dict[str, Any]],
) -> Optional[qdrant_models.Filter]:
    """Convert supported equality filters into a Qdrant filter."""
    if not filters:
        return None

    conditions: list[qdrant_models.Condition] = []
    for key, value in filters.items():
        if key not in _PAYLOAD_FIELDS:
            raise ValueError(f"Unsupported Qdrant filter field: {key}")
        conditions.append(
            qdrant_models.FieldCondition(
                key=key,
                match=qdrant_models.MatchValue(value=_serialize_filter_value(key, value)),
            )
        )

    return qdrant_models.Filter(must=conditions)


def _serialize_filter_value(key: str, value: Any) -> Union[bool, int, str]:
    """Return a scalar Qdrant match value for an equality filter."""
    if isinstance(value, UUID):
        return str(value)
    if key == "chunk_index":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Qdrant chunk_index filter must be an integer")
        return value
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported Qdrant filter value for {key}: {value!r}")


def _optional_str(value: Any, field_name: str) -> Optional[str]:
    """Validate an optional string payload field."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Qdrant payload {field_name} must be a string")
    return value


def _optional_int(value: Any, field_name: str) -> Optional[int]:
    """Validate an optional integer payload field."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Qdrant payload {field_name} must be an integer")
    return value


__all__ = [
    "map_filters_to_qdrant_filter",
    "map_vector_payload_from_qdrant_payload",
    "map_vector_payload_to_qdrant_payload",
]
