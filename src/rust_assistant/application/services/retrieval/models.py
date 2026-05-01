"""Internal retrieval models shared by application use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID


@dataclass(slots=True, frozen=True)
class RetrievalRequest:
    """Parameters for retrieving canonical documentation chunks."""

    query: str
    retrieval_limit: int
    reranking_limit: int
    use_reranking: bool


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """One hydrated chunk returned by the retrieval pipeline."""

    chunk_id: UUID
    document_id: UUID
    title: str
    source_path: str
    url: str
    section: Optional[str]
    item_path: Optional[str]
    crate: Optional[str]
    item_type: Optional[str]
    rust_version: Optional[str]
    score: float
    text: str
