from types import TracebackType
from typing import Optional, Protocol, Type

from rust_assistant.application.ports.repositories.chunk_repository import ChunkRepository
from rust_assistant.application.ports.repositories.document_repository import DocumentRepository


class UnitOfWork(Protocol):
    """Transaction boundary that owns session lifetime and exposes session-bound repositories."""
    documents: DocumentRepository
    chunks: ChunkRepository

    async def __aenter__(self) -> "UnitOfWork":
        """Open a unit-of-work scope and bind repositories to one session."""
        ...

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Close the unit-of-work scope and dispose of repository session state."""
        ...


    async def commit(self) -> None:
        """Commit the transaction owned by this unit of work."""
        ...

    async def rollback(self) -> None:
        """Rollback the transaction owned by this unit of work."""
        ...
