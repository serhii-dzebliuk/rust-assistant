"""Ingest-domain entities and structured parsing models."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from rust_assistant.schemas.enums import Crate, ItemType


class SourceType(str, Enum):
    """
    High-level source layout used by adapter factory selection.

    Multiple crates can map to the same source type.
    """

    BOOK = "book"
    REFERENCE = "reference"
    CARGO = "cargo"
    RUSTDOC = "rustdoc"


class BlockType(str, Enum):
    """Normalized structured content blocks extracted from HTML."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    DEFINITION_TERM = "definition_term"
    DEFINITION_DESC = "definition_desc"


class StructuredBlock(BaseModel):
    """Structured representation of a parsed HTML content block."""

    block_type: BlockType
    text: str
    html_tag: str
    heading_level: Optional[int] = None
    list_depth: Optional[int] = None
    code_language: Optional[str] = None
    anchor: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Metadata attached to parsed documentation page records."""

    crate: Crate = Crate.UNKNOWN
    item_path: Optional[str] = None
    item_type: Optional[ItemType] = None
    rust_version: Optional[str] = None
    url: Optional[str] = None
    raw_html_path: Optional[str] = None
    breadcrumbs: Optional[list[str]] = None


class Document(BaseModel):
    """Parsed documentation page produced by the ingest parsing stage."""

    doc_id: str
    title: str
    source_path: str
    text: str
    structured_blocks: list[StructuredBlock] = Field(default_factory=list)
    metadata: DocumentMetadata

    @staticmethod
    def generate_id(source_path: str, title: str) -> str:
        """Generate a stable short document id."""

        content = f"{source_path}::{title}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ChunkMetadata(BaseModel):
    """Metadata stored alongside a document chunk."""

    crate: Crate
    item_path: Optional[str] = None
    item_type: Optional[ItemType] = None
    rust_version: Optional[str] = None
    url: Optional[str] = None
    section: Optional[str] = None
    section_path: Optional[list[str]] = None
    anchor: Optional[str] = None
    chunk_index: int
    start_char: int
    end_char: int
    doc_title: str
    doc_source_path: str


class Chunk(BaseModel):
    """Chunk of text ready for embedding and indexing."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: ChunkMetadata
    text_hash: Optional[str] = None

    @staticmethod
    def generate_id(doc_id: str, chunk_index: int) -> str:
        """Generate a stable chunk identifier."""

        content = f"{doc_id}::chunk_{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute a normalized text hash for deduplication."""

        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        """Ensure chunk text is not empty."""

        if not value or not value.strip():
            raise ValueError("Chunk text cannot be empty")
        return value

    def model_post_init(self, __context: Any) -> None:
        """Populate the deduplication hash when absent."""

        if self.text_hash is None:
            self.text_hash = self.compute_text_hash(self.text)

    def to_jsonl(self) -> str:
        """Serialize the chunk to a JSONL line."""

        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> Chunk:
        """Deserialize a chunk from a JSONL line."""

        return cls.model_validate_json(line)


DocumentList = list[Document]
ChunkList = list[Chunk]


__all__ = [
    "BlockType",
    "Chunk",
    "ChunkList",
    "ChunkMetadata",
    "Document",
    "DocumentList",
    "DocumentMetadata",
    "SourceType",
    "StructuredBlock",
]
