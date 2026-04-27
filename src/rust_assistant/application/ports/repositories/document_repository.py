from collections.abc import Sequence
from typing import Optional
from typing import Protocol

from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.value_objects.identifiers import DocumentId


class DocumentRepository(Protocol):
    async def add(self, document: Document) -> None:
        """Persist one canonical document without generating a new identity."""
        ...

    async def add_many(self, documents: Sequence[Document]) -> None:
        """Persist multiple canonical documents without generating new identities."""
        ...

    async def get(self, document_id: DocumentId) -> Optional[Document]:
        """Load one canonical document by business UUID."""
        ...

    async def delete_all(self) -> None:
        """Delete the whole knowledge base via documents and DB-level chunk cascade."""
        ...
