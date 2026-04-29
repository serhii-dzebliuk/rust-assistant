from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
from uuid import UUID


@dataclass(frozen=True)
class VectorPayload:
    document_id: UUID

    crate: Optional[str] = None
    item_type: Optional[str] = None
    source_path: Optional[str] = None
    item_path: Optional[str] = None
    rust_version: Optional[str] = None
    section_title: Optional[str] = None
    chunk_index: Optional[int] = None
    text_hash: Optional[str] = None


@dataclass(frozen=True)
class VectorPoint:
    chunk_id: UUID
    vector: list[float]
    payload: VectorPayload


@dataclass(frozen=True)
class VectorSearchHit:
    chunk_id: UUID
    score: float
    payload: VectorPayload


class VectorStorage(Protocol):
    async def recreate_collection(self) -> None:
        ...

    async def upsert_vectors(self, points: Sequence[VectorPoint]) -> None:
        ...

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        score_threshold: Optional[float] = None,
    ) -> list[VectorSearchHit]:
        ...
