"""HTML document parser adapter."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union

from bs4 import BeautifulSoup

from rust_assistant.application.dto.document_parse import (
    DocumentParseFailureReason,
    DocumentParseResult,
)
from rust_assistant.application.policies.ingest.document_metadata import (
    ParsedDocumentFacts,
    build_item_path,
    detect_item_type,
    source_path_from_raw,
    source_path_to_url,
)
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate

from .layouts.factory import get_layout
from .utils import detect_crate_from_path, map_to_source_type

logger = logging.getLogger(__name__)


class HtmlDocumentParser:
    """
    Parse a single HTML file into a normalized `Document`.

    The adapter owns file reads, BeautifulSoup parsing, and source-layout
    extraction. Pure metadata decisions are delegated to application policy.
    """

    def __init__(self, raw_data_dir: Union[Path, str]):
        """Initialize parser with raw data root path."""
        self.raw_data_dir = Path(raw_data_dir).resolve()
        self._rust_versions: dict[Crate, Optional[str]] = {}

    def parse_file(self, file_path: Path) -> DocumentParseResult:
        """Parse one HTML file into a structured parse result."""
        try:
            html_content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s", file_path, exc)
            return DocumentParseResult.failed(
                file_path=file_path,
                reason=DocumentParseFailureReason.READ_ERROR,
                message=str(exc),
            )

        try:
            soup = BeautifulSoup(html_content, "lxml")
            crate = detect_crate_from_path(file_path)
            layout = get_layout(map_to_source_type(crate))

            title = layout.extract_title(soup, file_path)
            if not title:
                return self._failure(
                    file_path,
                    DocumentParseFailureReason.MISSING_TITLE,
                    "No title found",
                )

            main = layout.select_main(soup)
            if main is None:
                return self._failure(
                    file_path,
                    DocumentParseFailureReason.MISSING_MAIN_CONTENT,
                    "No main content found",
                )

            cleaned_main = layout.clean_main(main)
            structured_blocks = layout.extract_blocks(cleaned_main)
            text = layout.extract_text(structured_blocks)
            if not text or not text.strip():
                return self._failure(
                    file_path,
                    DocumentParseFailureReason.EMPTY_TEXT,
                    "No extracted text",
                )

            source_path = source_path_from_raw(self.raw_data_dir, file_path)
            url = source_path_to_url(source_path, crate)
            if url is None:
                return self._failure(
                    file_path,
                    DocumentParseFailureReason.UNSUPPORTED_SOURCE,
                    "No canonical URL could be built",
                )

            facts = ParsedDocumentFacts(
                raw_data_dir=self.raw_data_dir,
                file_path=file_path,
                crate=crate,
                title=title,
                text=text,
                breadcrumbs=self._extract_breadcrumbs(soup),
                rustdoc_body_classes=self._extract_body_classes(soup),
            )
            return DocumentParseResult.success(
                Document(
                    source_path=source_path,
                    title=title,
                    text=text,
                    crate=crate,
                    url=url,
                    item_path=build_item_path(facts),
                    item_type=detect_item_type(facts),
                    rust_version=self._resolve_rust_version(crate),
                    structured_blocks=structured_blocks,
                )
            )
        except Exception as exc:
            logger.exception("Unexpected error parsing %s", file_path)
            return DocumentParseResult.failed(
                file_path=file_path,
                reason=DocumentParseFailureReason.UNEXPECTED_ERROR,
                message=str(exc),
            )

    def _failure(
        self,
        file_path: Path,
        reason: DocumentParseFailureReason,
        message: str,
    ) -> DocumentParseResult:
        """Log and return a structured parse failure."""
        logger.warning("%s in %s", message, file_path)
        return DocumentParseResult.failed(
            file_path=file_path,
            reason=reason,
            message=message,
        )

    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> tuple[str, ...]:
        """Extract breadcrumb trail when available."""
        nav = soup.select_one("nav.sub")
        if nav:
            items = tuple(
                a.get_text(" ", strip=True)
                for a in nav.find_all("a")
                if a.get_text(" ", strip=True)
            )
            if items:
                return items
        breadcrumbs = soup.select_one(".breadcrumbs")
        if breadcrumbs:
            items = tuple(
                a.get_text(" ", strip=True)
                for a in breadcrumbs.find_all("a")
                if a.get_text(" ", strip=True)
            )
            if items:
                return items
        return ()

    def _extract_body_classes(self, soup: BeautifulSoup) -> tuple[str, ...]:
        """Extract body classes as plain parser facts."""
        if soup.body is None:
            return ()
        return tuple(str(css_class) for css_class in soup.body.get("class", []))

    def _resolve_rust_version(self, crate: Crate) -> Optional[str]:
        """Resolve docs snapshot version for a crate with in-memory caching."""
        if crate in self._rust_versions:
            return self._rust_versions[crate]

        resolver = {
            Crate.BOOK: self._resolve_book_version,
            Crate.STD: self._resolve_std_version,
            Crate.CARGO: self._resolve_cargo_version,
        }.get(crate)

        version = resolver() if resolver else None
        self._rust_versions[crate] = version
        return version

    def _resolve_book_version(self) -> Optional[str]:
        """Resolve Rust version from Rust Book landing page."""
        index_path = self.raw_data_dir / Crate.BOOK.value / "index.html"
        if not index_path.exists():
            return None

        soup = BeautifulSoup(index_path.read_text(encoding="utf-8", errors="replace"), "lxml")
        text = soup.get_text(" ", strip=True)
        match = re.search(r"Rust (\d+\.\d+\.\d+)", text)
        return match.group(1) if match else None

    def _resolve_std_version(self) -> Optional[str]:
        """Resolve rustdoc snapshot version from std crate index page."""
        index_path = self.raw_data_dir / Crate.STD.value / "index.html"
        if not index_path.exists():
            return None

        soup = BeautifulSoup(index_path.read_text(encoding="utf-8", errors="replace"), "lxml")
        rustdoc_vars_meta = soup.select_one('meta[name="rustdoc-vars"]')
        if rustdoc_vars_meta:
            channel = rustdoc_vars_meta.get("data-channel")
            if channel:
                return str(channel)

        sidebar_version = soup.select_one(".sidebar-crate .version")
        if sidebar_version:
            text = sidebar_version.get_text(" ", strip=True)
            if text:
                return text
        return None

    def _resolve_cargo_version(self) -> Optional[str]:
        """Resolve Cargo docs snapshot version from changelog page."""
        changelog_path = self.raw_data_dir / Crate.CARGO.value / "CHANGELOG.html"
        if not changelog_path.exists():
            return None

        soup = BeautifulSoup(changelog_path.read_text(encoding="utf-8", errors="replace"), "lxml")
        heading = soup.select_one("main h2")
        if not heading:
            return None

        match = re.search(r"Cargo (\d+\.\d+)", heading.get_text(" ", strip=True))
        return match.group(1) if match else None
