"""Source document selection policy for ingest discovery."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from rust_assistant.domain.enums import Crate


DEFAULT_CRATES: tuple[Crate, ...] = (
    Crate.STD,
    Crate.BOOK,
    Crate.CARGO,
    Crate.REFERENCE,
)

BOOK_EXCLUDE_FILES = {"README.html", "SUMMARY.html", "title-page.html"}
STD_EXCLUDE_FILES = {"all.html"}
CARGO_EXCLUDE_FILES = {"CHANGELOG.html"}

EXCLUDE_PATTERNS = {
    "*.js",
    "*.css",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.svg",
    "*.gif",
    "*.ico",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.json",
    "*.txt",
    "*.md",
    ".nojekyll",
    "search-index*.js",
    "searchindex*.js",
    "sidebar-items*.js",
    "print.html",
    "toc.html",
}

EXCLUDE_DIRS = {
    "theme",
    "css",
    "fonts",
    "FontAwesome",
    "images",
    "img",
    ".git",
    ".venv",
    "__pycache__",
    "first-edition",
    "second-edition",
    "2018-edition",
}


@dataclass(slots=True, frozen=True)
class SourceDocumentCandidate:
    """Plain facts used to decide whether a raw file enters ingest."""

    crate: Crate
    name: str
    relative_path: str
    path_parts: tuple[str, ...]
    is_file: bool
    is_html_redirect: bool
    is_book_legacy_page: bool
    has_meaningful_main_content: bool


def is_source_document_selected(candidate: SourceDocumentCandidate) -> bool:
    """Return whether a raw source file should enter the ingest pipeline."""
    if any(excluded_dir in candidate.path_parts for excluded_dir in EXCLUDE_DIRS):
        return False
    if any(
        fnmatch.fnmatch(candidate.name, pattern)
        or fnmatch.fnmatch(candidate.relative_path, pattern)
        for pattern in EXCLUDE_PATTERNS
    ):
        return False
    if not candidate.is_file:
        return False
    if candidate.crate == Crate.BOOK and candidate.name in BOOK_EXCLUDE_FILES:
        return False
    if candidate.crate == Crate.CARGO and candidate.name in CARGO_EXCLUDE_FILES:
        return False
    if candidate.crate == Crate.STD and candidate.name in STD_EXCLUDE_FILES:
        return False
    if candidate.crate == Crate.REFERENCE and candidate.name.endswith("-redirect.html"):
        return False
    if candidate.is_html_redirect:
        return False
    if candidate.crate == Crate.BOOK and candidate.is_book_legacy_page:
        return False
    if candidate.crate != Crate.STD and not candidate.has_meaningful_main_content:
        return False
    return True
