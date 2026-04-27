"""Adapter for rustdoc-style documentation pages."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag

from .base_adapter import HtmlAdapter


class RustdocAdapter(HtmlAdapter):
    """
    Adapter for rustdoc-style pages (std/core/alloc/test crates).

    It extracts canonical item titles such as `std::alloc::Layout` and
    removes rustdoc UI metadata blocks from the main content before text
    extraction.
    """

    main_selectors = ("main.content", "main", "div#content")

    def extract_title(self, soup: BeautifulSoup, file_path: Path) -> str:
        """
        Extract canonical rustdoc title with fallback to HTML title.

        Args:
            soup: Parsed rustdoc HTML document.
            file_path: Source HTML file path.

        Returns:
            Canonical item path-like title, or normalized fallback title.
        """
        canonical_title = self._canonical_title(soup, file_path)
        if canonical_title:
            return canonical_title

        if soup.title:
            raw_title = soup.title.get_text(" ", strip=True)
            normalized_title = self._normalize_html_title(raw_title)
            if normalized_title:
                return normalized_title
        return file_path.stem

    def clean_main(self, root: Tag) -> Tag:
        """
        Remove rustdoc-specific info badges and portability metadata.

        Args:
            root: Selected main content tag.

        Returns:
            Cleaned main content tag.
        """
        root = super().clean_main(root)
        for node in root.select(".item-info, .stab, .portability"):
            node.decompose()
        return root

    def _canonical_title(self, soup: BeautifulSoup, file_path: Path) -> Optional[str]:
        """
        Build canonical rustdoc title from breadcrumbs, heading, and file hints.

        Args:
            soup: Parsed rustdoc HTML document.
            file_path: Source HTML file path.

        Returns:
            Canonical title when it can be constructed, else `None`.
        """
        breadcrumbs = [
            node.get_text(" ", strip=True)
            for node in soup.select(".rustdoc-breadcrumbs a")
            if node.get_text(" ", strip=True)
        ]

        h1_name = self._extract_item_name(soup)
        file_based_title = self._title_from_file_path(file_path)
        if file_based_title:
            if breadcrumbs and not file_based_title.startswith(f"{breadcrumbs[0]}::"):
                return "::".join([*breadcrumbs, file_based_title])
            return file_based_title

        if h1_name:
            return "::".join([*breadcrumbs, h1_name]) if breadcrumbs else h1_name

        if breadcrumbs:
            return "::".join(breadcrumbs)

        return None

    def _extract_item_name(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract pure rustdoc item name from main heading without UI labels.

        Args:
            soup: Parsed rustdoc HTML document.

        Returns:
            Item name text or `None` when heading is absent.
        """
        h1 = soup.select_one("main h1")
        if not h1:
            return None

        # rustdoc puts the actual item name in a span and appends UI text like
        # "Copy item path" in a button, so extracting the full h1 text is noisy.
        name_node = h1.find("span")
        if name_node:
            name = name_node.get_text("", strip=True)
            if name:
                return name

        text_nodes = [
            text.strip()
            for text in h1.find_all(string=True, recursive=False)
            if text.strip()
        ]
        return text_nodes[-1] if text_nodes else None

    def _normalize_html_title(self, raw_title: str) -> str:
        """
        Normalize a fallback HTML title from rustdoc page head.

        Args:
            raw_title: Raw title string from `<title>`.

        Returns:
            Title without rustdoc suffix like ` - Rust`.
        """
        if raw_title.endswith(" - Rust"):
            raw_title = raw_title[:-7].strip()
        return raw_title

    def _title_from_file_path(self, file_path: Path) -> Optional[str]:
        """
        Infer canonical titles for special rustdoc pages from file path.

        Args:
            file_path: Source rustdoc file path.

        Returns:
            Canonical title for crate root, primitives, keywords, and macros.
        """
        path_parts = list(file_path.parts)
        try:
            crate_index = path_parts.index("std")
        except ValueError:
            return None

        relative_parts = path_parts[crate_index + 1:]
        if not relative_parts:
            return "std"

        if relative_parts == ["index.html"]:
            return "std"

        stem = file_path.stem
        if stem.startswith("primitive."):
            return f"std::primitive::{stem.removeprefix('primitive.')}"
        if stem.startswith("keyword."):
            return f"std::keyword::{stem.removeprefix('keyword.')}"
        if stem.startswith("macro."):
            return f"std::{stem.removeprefix('macro.')}!"
        return None
