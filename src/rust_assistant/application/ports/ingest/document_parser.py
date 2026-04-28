"""Port for parsing raw source files into canonical documents."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rust_assistant.application.dto.document_parse import DocumentParseResult


class DocumentParser(Protocol):
    """Parse raw source files into canonical documents."""

    def parse_file(self, file_path: Path) -> DocumentParseResult:
        """Parse one file path into a structured parse result."""
        ...
