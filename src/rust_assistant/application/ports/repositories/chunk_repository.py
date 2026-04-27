from collections.abc import Sequence
from typing import Optional
from typing import Protocol

from rust_assistant.application.dto.chunk_context import ChunkContext
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.value_objects.identifiers import ChunkId


class ChunkRepository(Protocol):
    async def add(self, chunk: Chunk) -> None:
        """Persist one canonical chunk without generating a new identity."""
        ...

    async def add_many(self, chunks: Sequence[Chunk]) -> None:
        """Persist multiple canonical chunks without generating new identities."""
        ...

    async def get(self, chunk_id: ChunkId) -> Optional[Chunk]:
        """Load one canonical chunk by business UUID."""
        ...

    async def get_contexts(self, ids: Sequence[ChunkId]) -> Sequence[ChunkContext]:
        """Load chunk contexts in input order for future retrieval-oriented read flows."""
        ...
