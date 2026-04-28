"""Layout strategy for Cargo book pages."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from .base_layout import HtmlLayout


class CargoLayout(HtmlLayout):
    """HTML layout tuned for Cargo documentation pages."""

    main_selectors = ("main", "div#content")
    extra_remove_selectors = (".sidebar", ".pagetoc")

    def extract_title(self, soup: BeautifulSoup, file_path: Path) -> str:
        """Extract a title from Cargo docs with fallbacks."""
        if soup.title:
            return soup.title.get_text(" ", strip=True)
        h1 = soup.select_one("main h1, h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return file_path.stem
