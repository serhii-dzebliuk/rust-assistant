"""Document domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.domain.value_objects.identifiers import (
    DocumentId,
    build_document_id,
)
from rust_assistant.domain.value_objects.structured_blocks import StructuredBlock


@dataclass(slots=True, frozen=True)
class Document:
    """Parsed documentation page represented as a domain entity."""

    id: DocumentId = field(init=False)
    source_path: str
    title: str
    text: str
    crate: Crate
    url: str
    item_path: Optional[str] = None
    item_type: Optional[ItemType] = None
    rust_version: Optional[str] = None
    structured_blocks: tuple[StructuredBlock, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate required fields and normalize immutable collections."""

        source_path = self.source_path.replace("\\", "/").strip()
        if not source_path:
            raise ValueError("Document source_path cannot be empty")
        if not self.title or not self.title.strip():
            raise ValueError("Document title cannot be empty")
        if not self.text or not self.text.strip():
            raise ValueError("Document text cannot be empty")
        if not self.url or not self.url.strip():
            raise ValueError("Document url cannot be empty")

        object.__setattr__(self, "id", build_document_id(source_path))
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "structured_blocks", tuple(self.structured_blocks))
