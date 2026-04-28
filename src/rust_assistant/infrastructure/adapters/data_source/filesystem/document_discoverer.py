"""
Discover HTML documentation files from a configured raw documentation directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from bs4 import BeautifulSoup, Tag

from rust_assistant.application.policies.ingest.document_selection import (
    DEFAULT_CRATES,
    SourceDocumentCandidate,
    is_source_document_selected,
)
from rust_assistant.domain.enums import Crate

logger = logging.getLogger(__name__)


class RawDocsDocumentDiscoverer:
    """
    Discover HTML files that should be parsed into documents.

    Filesystem traversal and lightweight HTML inspection stay here; the
    selection decision is delegated to application policy.
    """

    MEANINGFUL_CONTENT_TAGS: tuple[str, ...] = ("p", "li", "pre", "table", "blockquote", "dl")
    MAIN_SELECTORS: tuple[str, ...] = ("main", "section.normal", "article", "div#content", "body")
    NOISE_SELECTORS: tuple[str, ...] = (
        "script",
        "style",
        "noscript",
        "nav",
        "aside",
        "header",
        "footer",
        ".sidebar",
        "#sidebar",
        "button",
    )

    def __init__(self, raw_data_dir: Union[Path, str]):
        """
        Initialize discoverer.
        """
        self.raw_data_dir = Path(raw_data_dir).resolve()
        if not self.raw_data_dir.exists():
            raise ValueError(f"Raw data directory does not exist: {raw_data_dir}")

    def discover(
        self,
        crates: Optional[list[Crate]] = None,
        limit: Optional[int] = None,
    ) -> list[Path]:
        """
        Discover HTML files that are eligible for parsing.

        Args:
            crates: Optional crate list. If `None`, default crates are used.
            limit: Optional global limit for discovered files.

        Returns:
            List of absolute HTML paths selected for parsing.

        Example:
            >>> discoverer = RawDocsDocumentDiscoverer(raw_data_dir)
            >>> files = discoverer.discover(crates=[Crate.STD], limit=100)
            >>> len(files) <= 100
            True
        """
        selected_crates = crates or list(DEFAULT_CRATES)
        logger.info(
            "Discovering HTML files for crates=%s, limit=%s",
            [crate.value for crate in selected_crates],
            limit,
        )

        html_files: list[Path] = []
        for crate in selected_crates:
            crate_dir = self.raw_data_dir / crate.value
            if not crate_dir.exists():
                logger.warning("Crate directory not found: %s", crate_dir)
                continue

            crate_files = self._discover_in_directory(crate_dir, crate)
            html_files.extend(crate_files)
            logger.info("Discovered %s files in crate=%s", len(crate_files), crate.value)

            if limit and len(html_files) >= limit:
                logger.info("Discovery reached limit=%s; truncating results", limit)
                html_files = html_files[:limit]
                break

        logger.info("Discovery complete: %s HTML files total", len(html_files))
        return html_files

    def _discover_in_directory(self, directory: Path, crate: Crate) -> list[Path]:
        """
        Discover crate HTML files by traversing a directory tree.

        Args:
            directory: Crate root directory under the configured raw docs root.
            crate: Crate identifier for crate-specific rules.

        Returns:
            List of discovered file paths for the crate.
        """
        html_files: list[Path] = []
        for path in directory.rglob("*.html"):
            if not is_source_document_selected(self._candidate_for(path, crate)):
                continue
            html_files.append(path)
        logger.debug("Found %s files in %s/", len(html_files), crate.value)
        return html_files

    def _candidate_for(self, path: Path, crate: Crate) -> SourceDocumentCandidate:
        """Build plain selection facts for one raw source path."""
        try:
            relative_path = path.relative_to(self.raw_data_dir).as_posix()
        except ValueError:
            relative_path = path.as_posix()

        return SourceDocumentCandidate(
            crate=crate,
            name=path.name,
            relative_path=relative_path,
            path_parts=tuple(path.parts),
            is_file=path.is_file(),
            is_html_redirect=self._is_html_redirect(path),
            is_book_legacy_page=crate == Crate.BOOK and self._is_book_legacy_page(path),
            has_meaningful_main_content=(
                True if crate == Crate.STD else self._has_meaningful_main_content(path)
            ),
        )

    def _read_head(self, path: Path, bytes_count: int) -> Optional[str]:
        """
        Read initial chunk of a file for lightweight content checks.

        Args:
            path: Path to source HTML file.
            bytes_count: Number of characters to read from file start.

        Returns:
            Lower-cased head content or `None` when file cannot be read.
        """
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return handle.read(bytes_count).lower()
        except OSError:
            logger.debug("Could not read file during discovery check: %s", path)
            return None

    def _is_html_redirect(self, path: Path) -> bool:
        """
        Detect simple HTML redirect pages.

        Args:
            path: Candidate HTML file path.

        Returns:
            `True` for redirect-like pages, otherwise `False`.
        """
        head = self._read_head(path, 2048)
        if head is None:
            return False

        return (
            'http-equiv="refresh"' in head
            and "url=" in head
            and (
                "<title>redirection</title>" in head
                or "<title>redirecting...</title>" in head
                or "redirecting to..." in head
                or "window.location.replace(" in head
            )
        )

    def _is_book_legacy_page(self, path: Path) -> bool:
        """
        Detect legacy Rust Book alias pages.

        Args:
            path: Candidate HTML file path.

        Returns:
            `True` when file contains legacy marker text.
        """
        head = self._read_head(path, 4096)
        if head is None:
            return False
        legacy_marker = "there is a new edition of the book and this is an old link."
        return legacy_marker in head

    def _select_main_root(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        Select best effort main content root for lightweight content checks.

        Args:
            soup: Parsed HTML document.

        Returns:
            Main root `Tag` or `None` if no selectors match.
        """
        for selector in self.MAIN_SELECTORS:
            root = soup.select_one(selector)
            if isinstance(root, Tag):
                return root
        return None

    def _remove_noise_nodes(self, root: Tag) -> None:
        """
        Remove known layout and navigation noise from a content root.

        Args:
            root: Root tag selected for meaningful-content checks.
        """
        for selector in self.NOISE_SELECTORS:
            for node in root.select(selector):
                node.decompose()

    def _has_meaningful_main_content(self, path: Path) -> bool:
        """
        Check whether an HTML page has meaningful content beyond heading shells.

        Args:
            path: Candidate HTML file path.

        Returns:
            `True` when page appears to contain useful parseable content.
        """
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                soup = BeautifulSoup(handle.read(), "lxml")
        except OSError:
            logger.debug("Could not read file during content check: %s", path)
            return True

        root = self._select_main_root(soup)
        if not isinstance(root, Tag):
            return False
        self._remove_noise_nodes(root)

        if any(root.find(tag) for tag in self.MEANINGFUL_CONTENT_TAGS):
            return True

        # Heading-only pages are considered non-meaningful for non-rustdoc crates.
        return False
