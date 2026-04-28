
from qdrant_client import AsyncQdrantClient


class QdrantVectorStorage():
    
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        vector_size: int,
        distance: str,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance
