"""Adapter for Cargo book pages."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from .base_adapter import HtmlAdapter


class CargoAdapter(HtmlAdapter):
    """
    HTML adapter tuned for Cargo documentation pages.

    Cargo docs use mdBook-like structure but include Cargo-specific sidebars
    and page table-of-contents blocks that should be removed from parsed text.
    """

    main_selectors = ("main", "div#content")
    extra_remove_selectors = (".sidebar", ".pagetoc")

    def extract_title(self, soup: BeautifulSoup, file_path: Path) -> str:
        """
        Extract a title from Cargo docs with fallbacks.

        Args:
            soup: Parsed HTML document.
            file_path: Source HTML file path.

        Returns:
            Title from `<title>`, then `<h1>`, then file stem.
        """
        if soup.title:
            return soup.title.get_text(" ", strip=True)
        h1 = soup.select_one("main h1, h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return file_path.stem
