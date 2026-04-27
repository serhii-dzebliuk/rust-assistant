"""Domain identity types and deterministic UUID derivation rules.

Rules:
- `DocumentId = uuid5(DOCUMENT_ID_NAMESPACE, source_path)`
- `ChunkId = uuid5(CHUNK_ID_NAMESPACE, f"{document_id}:{chunk_index}")`
"""

from __future__ import annotations

from typing import NewType
from uuid import UUID, uuid5


DocumentId = NewType("DocumentId", UUID)
ChunkId = NewType("ChunkId", UUID)

DOCUMENT_ID_NAMESPACE = UUID("3d73b6b9-a8cb-4da8-b4b0-5ecdb09b3836")
CHUNK_ID_NAMESPACE = UUID("82f00626-72d1-4ee7-90a5-c281a4f3f4ec")


def build_document_id(source_path: str) -> DocumentId:
    """Build a stable document identity from the canonical source path."""

    return DocumentId(uuid5(DOCUMENT_ID_NAMESPACE, source_path))


def build_chunk_id(document_id: DocumentId, chunk_index: int) -> ChunkId:
    """Build a stable chunk identity from the parent document and chunk index."""

    return ChunkId(uuid5(CHUNK_ID_NAMESPACE, f"{document_id}:{chunk_index}"))
