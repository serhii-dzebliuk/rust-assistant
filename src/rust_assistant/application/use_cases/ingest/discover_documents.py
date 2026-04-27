"""Use case for discovering raw ingest source documents."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rust_assistant.application.ports.source_document_discovery import (
    SourceDocumentDiscoveryPort,
)
from rust_assistant.domain.enums import Crate


class DiscoverDocuments:
    """Discover raw documentation files for a selected crate scope."""

    def __init__(self, discovery: SourceDocumentDiscoveryPort):
        self._discovery = discovery

    def execute(
        self,
        *,
        crates: Optional[list[Crate]] = None,
        limit: Optional[int] = None,
    ) -> list[Path]:
        """Return discovered raw HTML files for the requested ingest run."""
        return self._discovery.discover(crates=crates, limit=limit)
