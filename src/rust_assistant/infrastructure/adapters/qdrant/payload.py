"""Qdrant payload builders for chunk vector synchronization."""

from __future__ import annotations

from typing import Any

from rust_assistant.infrastructure.adapters.sqlalchemy.models import ChunkRecord


def build_chunk_payload(chunk: ChunkRecord) -> dict[str, Any]:
    """
    Build the minimal Qdrant payload for a persisted chunk.

    The payload intentionally excludes canonical chunk text. Qdrant owns vector
    search metadata only; PostgreSQL remains the source of truth for content.
    """
    document = chunk.document
    if document is None:
        raise ValueError("ChunkRecord.document must be loaded to build Qdrant payload")

    return {
        "chunk_id": str(chunk.id),
        "document_id": str(document.id),
        "crate": document.crate,
        "item_type": document.item_type,
        "rust_version": document.rust_version,
        "source_path": document.source_path,
        "item_path": document.item_path,
        "chunk_index": chunk.chunk_index,
        "hash": chunk.hash,
    }


__all__ = ["build_chunk_payload"]
