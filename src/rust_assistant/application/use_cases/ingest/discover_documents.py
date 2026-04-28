"""Use case for discovering raw ingest source documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rust_assistant.application.ports.ingest.document_discoverer import (
    DocumentDiscoverer,
)
from rust_assistant.domain.enums import Crate


@dataclass(slots=True, frozen=True)
class DiscoverDocumentsCommand:
    """Input for discovering raw documentation files."""

    crates: Optional[list[Crate]] = None
    limit: Optional[int] = None


@dataclass(slots=True, frozen=True)
class DiscoverDocumentsResult:
    """Discovered raw documentation files."""

    discovered_files: list[Path]


class DiscoverDocumentsUseCase:
    """Discover raw documentation files for a selected crate scope."""

    def __init__(self, discovery: DocumentDiscoverer):
        self._discovery = discovery

    def execute(self, command: DiscoverDocumentsCommand) -> DiscoverDocumentsResult:
        """Return discovered raw HTML files for the requested ingest run."""
        return DiscoverDocumentsResult(
            discovered_files=self._discovery.discover(
                crates=command.crates,
                limit=command.limit,
            )
        )
