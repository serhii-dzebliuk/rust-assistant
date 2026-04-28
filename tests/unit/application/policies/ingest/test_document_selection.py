import pytest

from rust_assistant.application.policies.ingest.document_selection import (
    SourceDocumentCandidate,
    is_source_document_selected,
)
from rust_assistant.domain.enums import Crate

pytestmark = pytest.mark.unit


def candidate(**overrides):
    values = {
        "crate": Crate.STD,
        "name": "index.html",
        "relative_path": "std/index.html",
        "path_parts": ("rust-docs", "std", "index.html"),
        "is_file": True,
        "is_html_redirect": False,
        "is_book_legacy_page": False,
        "has_meaningful_main_content": True,
    }
    values.update(overrides)
    return SourceDocumentCandidate(**values)


def test_source_document_selection_accepts_parseable_std_html():
    assert is_source_document_selected(candidate()) is True


@pytest.mark.parametrize(
    "selected_candidate",
    [
        candidate(path_parts=("rust-docs", "std", "theme", "index.html")),
        candidate(name="all.html"),
        candidate(name="redirect.html", is_html_redirect=True),
        candidate(crate=Crate.REFERENCE, name="foo-redirect.html"),
        candidate(crate=Crate.BOOK, name="README.html"),
        candidate(crate=Crate.BOOK, name="chapter.html", is_book_legacy_page=True),
        candidate(crate=Crate.CARGO, name="CHANGELOG.html"),
        candidate(crate=Crate.BOOK, name="empty.html", has_meaningful_main_content=False),
        candidate(is_file=False),
    ],
)
def test_source_document_selection_rejects_ineligible_files(selected_candidate):
    assert is_source_document_selected(selected_candidate) is False
