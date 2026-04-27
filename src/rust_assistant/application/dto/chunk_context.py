"""Read DTOs used by retrieval-oriented use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.domain.value_objects.identifiers import ChunkId, DocumentId


@dataclass(slots=True, frozen=True)
class ChunkContext:
    """Canonical chunk content and document metadata loaded for retrieval."""

    chunk_id: ChunkId
    document_id: DocumentId
    text: str
    title: str
    source_path: str
    url: str
    section_title: Optional[str]
    section_path: tuple[str, ...]
    section_anchor: Optional[str]
    item_path: Optional[str]
    crate: Crate
    item_type: Optional[ItemType]
    rust_version: Optional[str]
    chunk_index: int
