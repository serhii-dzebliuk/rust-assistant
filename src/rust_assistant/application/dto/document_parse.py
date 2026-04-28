"""Application DTOs for document parsing results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from rust_assistant.domain.entities.documents import Document


class DocumentParseFailureReason(str, Enum):
    """Reasons a raw source file could not be parsed into a document."""

    READ_ERROR = "read_error"
    MISSING_TITLE = "missing_title"
    MISSING_MAIN_CONTENT = "missing_main_content"
    EMPTY_TEXT = "empty_text"
    UNSUPPORTED_SOURCE = "unsupported_source"
    UNEXPECTED_ERROR = "unexpected_error"


@dataclass(slots=True, frozen=True)
class DocumentParseFailure:
    """Structured parse failure reported by document parser adapters."""

    file_path: Path
    reason: DocumentParseFailureReason
    message: str


@dataclass(slots=True, frozen=True)
class DocumentParseResult:
    """Result of parsing one raw source file."""

    document: Optional[Document] = None
    failure: Optional[DocumentParseFailure] = None

    def __post_init__(self) -> None:
        """Ensure parse results are either success or failure, never both."""
        if (self.document is None) == (self.failure is None):
            raise ValueError("DocumentParseResult must contain exactly one outcome")

    @classmethod
    def success(cls, document: Document) -> "DocumentParseResult":
        """Build a successful parse result."""
        return cls(document=document)

    @classmethod
    def failed(
        cls,
        *,
        file_path: Path,
        reason: DocumentParseFailureReason,
        message: str,
    ) -> "DocumentParseResult":
        """Build a failed parse result."""
        return cls(
            failure=DocumentParseFailure(
                file_path=file_path,
                reason=reason,
                message=message,
            )
        )

