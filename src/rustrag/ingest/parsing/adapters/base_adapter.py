"""Base adapter contract for source-specific HTML parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from ..core import COMMON_REMOVE_SELECTORS, extract_structured_text, remove_noise


class HtmlAdapter(ABC):
    """
    Base adapter that defines a common HTML parsing contract.

    Source-specific adapters override selectors and title extraction to
    support different documentation layouts (book, reference, rustdoc).
    """

    main_selectors: tuple[str, ...] = ("main", "body")
    extra_remove_selectors: tuple[str, ...] = ()

    def select_main(self, soup: BeautifulSoup) -> Tag | None:
        """
        Select the most relevant main-content node from a parsed document.

        Args:
            soup: Parsed HTML document.

        Returns:
            Selected main content tag or `None` when no candidate exists.

        Example:
            >>> adapter.select_main(soup)
            <Tag ...>
        """
        for selector in self.main_selectors:
            node = soup.select_one(selector)
            if isinstance(node, Tag):
                return node
        body = soup.body
        return body if isinstance(body, Tag) else None

    def clean_main(self, root: Tag) -> Tag:
        """
        Remove common and adapter-specific noisy nodes from main content.

        Args:
            root: Main content tag selected for extraction.

        Returns:
            Mutated root tag with noisy nodes removed.
        """
        remove_noise(root, COMMON_REMOVE_SELECTORS + self.extra_remove_selectors)
        return root

    @abstractmethod
    def extract_title(self, soup: BeautifulSoup, file_path: Path) -> str:
        """
        Extract a source-specific title for the parsed document.

        Args:
            soup: Parsed HTML document.
            file_path: Source HTML file path.

        Returns:
            Stable title string used in metadata and document id generation.
        """
        raise NotImplementedError

    def extract_text(self, root: Tag) -> str:
        """
        Convert cleaned HTML main content into normalized text blocks.

        Args:
            root: Cleaned main content tag.

        Returns:
            Structured, normalized text extracted from the HTML subtree.
        """
        return extract_structured_text(root)
