"""
Data models for RustRAG pipeline.
Defines contracts for Documents, Chunks, and Search Results.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class ItemType(str, Enum):
    """Type of Rust documentation item."""
    FUNCTION = "fn"
    STRUCT = "struct"
    TRAIT = "trait"
    METHOD = "method"
    IMPL = "impl"
    MODULE = "module"
    MACRO = "macro"
    ENUM = "enum"
    CONSTANT = "constant"
    TYPE_ALIAS = "type"
    BOOK_SECTION = "book_section"
    UNKNOWN = "unknown"


class Crate(str, Enum):
    """Known Rust crates/documentation sources."""
    STD = "std"
    CORE = "core"
    ALLOC = "alloc"
    PROC_MACRO = "proc_macro"
    TEST = "test"
    BOOK = "book"
    REFERENCE = "reference"
    RUSTC = "rustc"
    CARGO = "cargo"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata for a Rust documentation document."""
    crate: Crate = Crate.UNKNOWN
    item_path: Optional[str] = None  # e.g., "std::vec::Vec::push"
    item_type: ItemType = ItemType.UNKNOWN
    rust_version: Optional[str] = None
    url: Optional[str] = None
    raw_html_path: Optional[str] = None  # path to original HTML file
    scraped_at: Optional[datetime] = None

    # Additional fields for filtering/debugging
    breadcrumbs: Optional[list[str]] = None
    parent_module: Optional[str] = None


class Document(BaseModel):
    """
    Represents a parsed Rust documentation page.
    Output of parsing stage (ingest/parse_html.py).
    """
    doc_id: str
    title: str
    source_path: str  # relative path from data/raw/
    text: str  # cleaned main content
    metadata: DocumentMetadata

    @staticmethod
    def generate_id(source_path: str, title: str) -> str:
        """Generate unique document ID from source path and title."""
        content = f"{source_path}::{title}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Ensure text is not empty."""
        if not v or not v.strip():
            raise ValueError("Document text cannot be empty")
        return v

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> Document:
        """Deserialize from JSONL format."""
        return cls.model_validate_json(line)


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    # Inherited from Document
    crate: Crate
    item_path: Optional[str] = None
    item_type: ItemType
    rust_version: Optional[str] = None
    url: Optional[str] = None

    # Chunk-specific
    section: Optional[str] = None  # e.g., "Examples", "Panics", "Safety"
    anchor: Optional[str] = None  # HTML anchor for direct linking
    chunk_index: int  # position in document (0-indexed)
    start_char: int  # start position in original document text
    end_char: int  # end position in original document text

    # Document context
    doc_title: str
    doc_source_path: str


class Chunk(BaseModel):
    """
    Represents a chunk of text ready for embedding and indexing.
    Output of chunking stage (ingest/chunk.py).
    """
    chunk_id: str
    doc_id: str  # parent document ID
    text: str  # chunk text (900-1200 chars for MVP)
    metadata: ChunkMetadata

    # Optional fields for deduplication
    text_hash: Optional[str] = None

    @staticmethod
    def generate_id(doc_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{doc_id}::chunk_{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute hash for deduplication (normalized text)."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Ensure chunk text is not empty."""
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Compute text hash after initialization if not provided."""
        if self.text_hash is None:
            self.text_hash = self.compute_text_hash(self.text)

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> Chunk:
        """Deserialize from JSONL format."""
        return cls.model_validate_json(line)


class SearchResult(BaseModel):
    """
    Represents a search result from vector database.
    Used in retrieval and API responses.
    """
    chunk: Chunk
    score: float = Field(..., ge=0.0, le=1.0)  # relevance score 0-1
    snippet: Optional[str] = None  # formatted snippet for display (with highlights)
    rank: Optional[int] = None  # position in results (1-indexed)

    @field_validator('score')
    @classmethod
    def score_in_range(cls, v: float) -> float:
        """Ensure score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v

    def format_snippet(self, max_length: int = 200, context_chars: int = 50) -> str:
        """
        Format chunk text as snippet for display.

        Args:
            max_length: Maximum snippet length
            context_chars: Characters of context around match
        """
        text = self.chunk.text
        if len(text) <= max_length:
            return text

        # Simple truncation for MVP (can improve with highlighting later)
        return text[:max_length - 3] + "..."

    def model_post_init(self, __context: Any) -> None:
        """Generate snippet if not provided."""
        if self.snippet is None:
            self.snippet = self.format_snippet()


class QueryRequest(BaseModel):
    """Request model for search/chat endpoints."""
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None  # e.g., {"crate": "std", "item_type": "fn"}

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Ensure query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchResponse(BaseModel):
    """Response model for /search endpoint."""
    results: list[SearchResult]
    query: str
    total_results: int
    filters_applied: Optional[Dict[str, Any]] = None
    retrieval_time_ms: Optional[float] = None


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None
    debug: bool = Field(default=False)

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Ensure question is not empty."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    answer: str  # markdown formatted answer
    sources: list[SearchResult]
    question: str

    # Optional debug info
    debug_info: Optional[Dict[str, Any]] = None
    # debug_info can contain:
    # - retrieval_time_ms
    # - generation_time_ms
    # - top_k_scores
    # - filters_applied
    # - prompt_used
    # - model_name

    confidence: Optional[str] = None  # "high", "medium", "low", "unknown"


# Type aliases for convenience
DocumentList = list[Document]
ChunkList = list[Chunk]
SearchResults = list[SearchResult]
