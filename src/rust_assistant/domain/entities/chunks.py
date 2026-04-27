"""Chunk domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Optional

from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.domain.errors import InvalidChunkTextError
from rust_assistant.domain.value_objects.identifiers import (
    ChunkId,
    DocumentId,
    build_chunk_id,
    build_document_id,
)


@dataclass(slots=True, frozen=True)
class Chunk:
    """Chunk of text ready for embedding and indexing."""

    id: ChunkId = field(init=False)
    document_id: DocumentId = field(init=False)
    source_path: str
    chunk_index: int
    text: str
    crate: Crate
    start_offset: int
    end_offset: int
    section_path: tuple[str, ...] = field(default_factory=tuple)
    section_anchor: Optional[str] = None
    item_path: Optional[str] = None
    item_type: Optional[ItemType] = None
    rust_version: Optional[str] = None
    url: Optional[str] = None
    token_count: Optional[int] = None
    text_hash: str = ""

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute a normalized text hash for deduplication."""

        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    @property
    def section_title(self) -> Optional[str]:
        """Return the most specific section heading when available."""

        if not self.section_path:
            return None
        return self.section_path[-1]

    def __post_init__(self) -> None:
        """Validate fields and populate derived values."""

        source_path = self.source_path.replace("\\", "/").strip()
        if not source_path:
            raise ValueError("Chunk source_path cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("Chunk chunk_index cannot be negative")
        if self.start_offset < 0 or self.end_offset < 0:
            raise ValueError("Chunk offsets cannot be negative")
        if self.end_offset < self.start_offset:
            raise ValueError("Chunk end_offset cannot be smaller than start_offset")
        if not self.text or not self.text.strip():
            raise InvalidChunkTextError("Chunk text cannot be empty")

        document_id = build_document_id(source_path)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "id", build_chunk_id(document_id, self.chunk_index))
        object.__setattr__(self, "section_path", tuple(self.section_path))
        if not self.text_hash:
            object.__setattr__(self, "text_hash", self.compute_text_hash(self.text))
