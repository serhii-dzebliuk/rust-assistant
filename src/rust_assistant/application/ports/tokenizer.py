"""Port for discovering raw source documents for ingest."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from rust_assistant.domain.enums import Crate


class SourceDocumentDiscoveryPort(Protocol):
    """Discover source documents that should enter the ingest pipeline."""

    def discover(
        self,
        *,
        crates: Optional[list[Crate]] = None,
        limit: Optional[int] = None,
    ) -> list[Path]:
        """Return discovered raw document paths for the requested crate scope."""
        ...
