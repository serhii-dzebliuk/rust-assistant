from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from rustrag.ingest.parsing.adapters.factory import get_adapter
from rustrag.ingest.parsing.utils import (
    detect_crate_from_path,
    map_to_source_type,
    source_path_from_raw,
    source_path_to_url,
)
from rustrag.models import ItemType, DocumentMetadata, Document, Crate

logger = logging.getLogger(__name__)

class PageParser:
    """Parser orchestrator with source-specific adapters."""

    ITEM_TYPE_PATTERNS = {
        ItemType.FUNCTION: (r"\bfn\b", r"\bfunction\b"),
        ItemType.STRUCT: (r"\bstruct\b",),
        ItemType.TRAIT: (r"\btrait\b",),
        ItemType.METHOD: (r"\bmethod\b",),
        ItemType.IMPL: (r"\bimpl\b", r"\bimplementation\b"),
        ItemType.MODULE: (r"\bmod\b", r"\bmodule\b"),
        ItemType.MACRO: (r"\bmacro\b",),
        ItemType.ENUM: (r"\benum\b",),
        ItemType.CONSTANT: (r"\bconst\b", r"\bconstant\b"),
        ItemType.TYPE_ALIAS: (r"\btype\b",),
    }
    RUSTDOC_FILE_PREFIX_TO_ITEM_TYPE = {
        "fn.": ItemType.FUNCTION,
        "struct.": ItemType.STRUCT,
        "trait.": ItemType.TRAIT,
        "macro.": ItemType.MACRO,
        "enum.": ItemType.ENUM,
        "constant.": ItemType.CONSTANT,
        "type.": ItemType.TYPE_ALIAS,
        "mod.": ItemType.MODULE,
    }
    RUSTDOC_BODY_CLASS_TO_ITEM_TYPE = {
        "mod": ItemType.MODULE,
        "struct": ItemType.STRUCT,
        "trait": ItemType.TRAIT,
        "macro": ItemType.MACRO,
        "enum": ItemType.ENUM,
        "constant": ItemType.CONSTANT,
        "type": ItemType.TYPE_ALIAS,
        "function": ItemType.FUNCTION,
        "fn": ItemType.FUNCTION,
        "method": ItemType.METHOD,
        "impl": ItemType.IMPL,
    }

    def __init__(self, raw_data_dir: Path | str):
        self.raw_data_dir = Path(raw_data_dir).resolve()
        self._rust_versions: dict[Crate, str | None] = {}

    def parse_file(self, file_path: Path) -> Optional[Document]:
        try:
            html_content = file_path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "lxml")
            crate = detect_crate_from_path(file_path)
            adapter = get_adapter(map_to_source_type(crate))

            title = adapter.extract_title(soup, file_path)
            if not title:
                logger.warning("No title found in %s", file_path)
                return None

            main = adapter.select_main(soup)
            if main is None:
                logger.warning("No main content found in %s", file_path)
                return None

            cleaned_main = adapter.clean_main(main)
            text = adapter.extract_text(cleaned_main)
            if not text or not text.strip():
                logger.warning("No extracted text in %s", file_path)
                return None

            source_path = source_path_from_raw(self.raw_data_dir, file_path)
            item_path = self._extract_item_path(soup, file_path, crate, title)
            item_type = self._detect_item_type(title, text, crate, file_path, soup)
            breadcrumbs = self._extract_breadcrumbs(soup)
            rust_version = self._resolve_rust_version(crate)
            url = source_path_to_url(source_path, crate)

            metadata = DocumentMetadata(
                crate=crate,
                item_path=item_path,
                item_type=item_type,
                rust_version=rust_version,
                url=url,
                raw_html_path=str(file_path),
                breadcrumbs=breadcrumbs,
            )
            return Document(
                doc_id=Document.generate_id(source_path, title),
                title=title,
                source_path=source_path,
                text=text,
                metadata=metadata,
            )
        except Exception as exc:
            logger.error("Error parsing %s: %s", file_path, exc, exc_info=True)
            return None

    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> Optional[list[str]]:
        nav = soup.select_one("nav.sub")
        if nav:
            items = [a.get_text(" ", strip=True) for a in nav.find_all("a")]
            if items:
                return items
        breadcrumbs = soup.select_one(".breadcrumbs")
        if breadcrumbs:
            items = [a.get_text(" ", strip=True) for a in breadcrumbs.find_all("a")]
            if items:
                return items
        return None

    def _extract_item_path(self, soup: BeautifulSoup, file_path: Path, crate: Crate, title: str) -> Optional[str]:
        if crate in {Crate.STD, Crate.CORE, Crate.ALLOC, Crate.PROC_MACRO, Crate.TEST}:
            return title or None

        breadcrumbs = self._extract_breadcrumbs(soup)
        if breadcrumbs:
            return "::".join(breadcrumbs)

        try:
            rel = file_path.resolve().relative_to((self.raw_data_dir / crate.value).resolve())
            parts = list(rel.parts)
            if not parts:
                return None
            parts[-1] = rel.stem
            parts = [p for p in parts if p not in {"index", "."}]
            if not parts:
                return crate.value
            return f"{crate.value}::" + "::".join(parts)
        except ValueError:
            return None

    def _detect_item_type(
        self,
        title: str,
        text: str,
        crate: Crate,
        file_path: Path,
        soup: BeautifulSoup,
    ) -> ItemType | None:
        if crate != Crate.STD:
            return None

        rustdoc_item_type = self._detect_rustdoc_item_type(title, file_path, soup)
        if rustdoc_item_type is not None:
            return rustdoc_item_type

        title_l = title.lower()

        for item_type, patterns in self.ITEM_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title_l, re.IGNORECASE):
                    return item_type

        sample = text[:700].lower()
        for item_type, patterns in self.ITEM_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sample, re.IGNORECASE):
                    return item_type
        return ItemType.UNKNOWN

    def _detect_rustdoc_item_type(self, title: str, file_path: Path, soup: BeautifulSoup) -> ItemType | None:
        stem = file_path.stem
        for prefix, item_type in self.RUSTDOC_FILE_PREFIX_TO_ITEM_TYPE.items():
            if stem.startswith(prefix):
                return item_type

        if file_path.name == "index.html":
            return ItemType.MODULE

        body = soup.body
        if body:
            for css_class in body.get("class", []):
                if css_class in self.RUSTDOC_BODY_CLASS_TO_ITEM_TYPE:
                    return self.RUSTDOC_BODY_CLASS_TO_ITEM_TYPE[css_class]

        title_l = title.lower()

        if title_l == "std":
            return ItemType.MODULE

        item_name = title.rsplit("::", 1)[-1]
        if item_name.endswith("!"):
            return ItemType.MACRO
        if "::keyword::" in title_l or "::primitive::" in title_l:
            return ItemType.UNKNOWN
        return None

    def _resolve_rust_version(self, crate: Crate) -> str | None:
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

    def _resolve_book_version(self) -> str | None:
        index_path = self.raw_data_dir / Crate.BOOK.value / "index.html"
        if not index_path.exists():
            return None

        soup = BeautifulSoup(index_path.read_text(encoding="utf-8", errors="replace"), "lxml")
        text = soup.get_text(" ", strip=True)
        match = re.search(r"Rust (\d+\.\d+\.\d+)", text)
        return match.group(1) if match else None

    def _resolve_std_version(self) -> str | None:
        index_path = self.raw_data_dir / Crate.STD.value / "index.html"
        if not index_path.exists():
            return None

        soup = BeautifulSoup(index_path.read_text(encoding="utf-8", errors="replace"), "lxml")
        vars_meta = soup.select_one('meta[name="rustdoc-vars"]')
        if vars_meta:
            channel = vars_meta.get("data-channel")
            if channel:
                return channel

        sidebar_version = soup.select_one(".sidebar-crate .version")
        if sidebar_version:
            text = sidebar_version.get_text(" ", strip=True)
            if text:
                return text
        return None

    def _resolve_cargo_version(self) -> str | None:
        changelog_path = self.raw_data_dir / Crate.CARGO.value / "CHANGELOG.html"
        if not changelog_path.exists():
            return None

        soup = BeautifulSoup(changelog_path.read_text(encoding="utf-8", errors="replace"), "lxml")
        heading = soup.select_one("main h2")
        if not heading:
            return None

        match = re.search(r"Cargo (\d+\.\d+)", heading.get_text(" ", strip=True))
        return match.group(1) if match else None
