"""Base layout contract for source-specific HTML parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag

from rust_assistant.domain.value_objects.structured_blocks import StructuredBlock

from ..core import (
    COMMON_REMOVE_SELECTORS,
    blocks_to_text,
    extract_structured_blocks,
    remove_noise,
)


class HtmlLayout(ABC):
    """
    Base layout that defines a common HTML parsing contract.

    Source-specific layouts override selectors and title extraction to support
    different documentation layouts.
    """

    main_selectors: tuple[str, ...] = ("main", "body")
    extra_remove_selectors: tuple[str, ...] = ()

    def select_main(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Select the most relevant main-content node from a parsed document."""
        for selector in self.main_selectors:
            node = soup.select_one(selector)
            if isinstance(node, Tag):
                return node
        body = soup.body
        return body if isinstance(body, Tag) else None

    def clean_main(self, root: Tag) -> Tag:
        """Remove common and layout-specific noisy nodes from main content."""
        remove_noise(root, COMMON_REMOVE_SELECTORS + self.extra_remove_selectors)
        return root

    @abstractmethod
    def extract_title(self, soup: BeautifulSoup, file_path: Path) -> str:
        """Extract a source-specific title for the parsed document."""
        raise NotImplementedError

    def extract_blocks(self, root: Tag) -> list[StructuredBlock]:
        """Extract structured content blocks from cleaned main content."""
        return extract_structured_blocks(root)

    def extract_text(self, blocks: list[StructuredBlock]) -> str:
        """Convert structured content blocks into normalized text."""
        return blocks_to_text(blocks)
