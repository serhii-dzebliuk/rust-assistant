"""Qdrant implementation of the vector-storage port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from rust_assistant.application.ports.vector_storage import (
    VectorPoint,
    VectorSearchHit,
)
from rust_assistant.infrastructure.adapters.vector_storage.qdrant.mappers import (
    map_vector_payload_from_qdrant_payload,
    map_vector_payload_to_qdrant_payload,
)


class QdrantVectorStorage:
    """Store and search vector embeddings in Qdrant."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        vector_size: int,
        distance: str,
        upsert_batch_size: int = 256,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = _resolve_distance(distance)
        if upsert_batch_size < 1:
            raise ValueError("Qdrant upsert_batch_size must be >= 1")
        self._upsert_batch_size = upsert_batch_size

    async def recreate_collection(self) -> None:
        """Recreate the configured Qdrant collection."""
        await self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self._vector_size,
                distance=self._distance,
            ),
        )

    async def upsert_vectors(self, points: Sequence[VectorPoint]) -> None:
        """Upsert vector points into the configured Qdrant collection."""
        if not points:
            return

        for batch_start in range(0, len(points), self._upsert_batch_size):
            batch = points[batch_start : batch_start + self._upsert_batch_size]
            await self._client.upsert(
                collection_name=self._collection_name,
                points=[_map_point_to_qdrant(point) for point in batch],
            )

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        score_threshold: Optional[float] = None,
    ) -> list[VectorSearchHit]:
        """Search for nearest vector matches in Qdrant."""
        result = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=None,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
        return [
            VectorSearchHit(
                chunk_id=_point_id_to_uuid(point.id),
                score=point.score,
                payload=map_vector_payload_from_qdrant_payload(point.payload or {}),
            )
            for point in result.points
        ]


def _resolve_distance(distance: str) -> qdrant_models.Distance:
    """Resolve a configured distance string to the Qdrant enum."""
    normalized_distance = distance.strip().lower()
    for candidate in qdrant_models.Distance:
        if candidate.name.lower() == normalized_distance:
            return candidate
        if candidate.value.lower() == normalized_distance:
            return candidate
    raise ValueError(f"Unsupported Qdrant distance: {distance}")


def _map_point_to_qdrant(point: VectorPoint) -> qdrant_models.PointStruct:
    """Convert an application vector point into a Qdrant point struct."""
    return qdrant_models.PointStruct(
        id=point.chunk_id,
        vector=point.vector,
        payload=map_vector_payload_to_qdrant_payload(point.payload),
    )


def _point_id_to_uuid(point_id: Union[int, str, UUID]) -> UUID:
    """Convert a Qdrant point id into the application chunk UUID."""
    if isinstance(point_id, int):
        raise ValueError("Qdrant point id must be a UUID-compatible string")
    try:
        return UUID(str(point_id))
    except ValueError as exc:
        raise ValueError("Qdrant point id must be a valid UUID") from exc
