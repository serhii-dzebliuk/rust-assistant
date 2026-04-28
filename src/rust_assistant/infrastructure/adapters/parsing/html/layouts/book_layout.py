"""Layout strategy for Rust Book (mdBook-like) pages."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from .base_layout import HtmlLayout


class BookLayout(HtmlLayout):
    """HTML layout tuned for Rust Book pages."""

    main_selectors = ("main", "div#content", "section.normal")
    extra_remove_selectors = (
        ".chapter .nav",
        ".nav-chapters",
        ".menu-bar",
        ".menu-title",
    )

    def extract_title(self, soup: BeautifulSoup, file_path: Path) -> str:
        """Extract a title from book HTML with graceful fallbacks."""
        if soup.title:
            return soup.title.get_text(" ", strip=True)
        h1 = soup.select_one("main h1, h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return file_path.stem
