"""Port for parsing raw source files into canonical documents."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from rust_assistant.domain.entities.documents import Document


class DocumentParserPort(Protocol):
    """Parse a raw source file into a canonical document."""

    def parse_file(self, file_path: Path) -> Optional[Document]:
        """Parse one file path into a document or return `None` on parse failure."""
        ...
