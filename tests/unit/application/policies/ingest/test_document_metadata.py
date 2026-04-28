from pathlib import Path

import pytest

from rust_assistant.application.policies.ingest.document_metadata import (
    ParsedDocumentFacts,
    build_item_path,
    detect_item_type,
    source_path_from_raw,
    source_path_to_url,
)
from rust_assistant.domain.enums import Crate, ItemType

pytestmark = pytest.mark.unit


def facts(**overrides):
    values = {
        "raw_data_dir": Path("rust-docs"),
        "file_path": Path("rust-docs/std/alloc/struct.Layout.html"),
        "crate": Crate.STD,
        "title": "std::alloc::Layout",
        "text": "pub struct Layout",
    }
    values.update(overrides)
    return ParsedDocumentFacts(**values)


def test_metadata_builds_source_path_and_url():
    source_path = source_path_from_raw(
        Path("rust-docs"),
        Path("rust-docs/book/ch01-03-hello-cargo.html"),
    )

    assert source_path == "book/ch01-03-hello-cargo.html"
    assert (
        source_path_to_url(source_path, Crate.BOOK)
        == "https://doc.rust-lang.org/book/ch01-03-hello-cargo.html"
    )


def test_metadata_builds_page_item_path_from_relative_file_path():
    item_path = build_item_path(
        facts(
            file_path=Path("rust-docs/cargo/commands/cargo-build.html"),
            crate=Crate.CARGO,
            title="cargo build",
        )
    )

    assert item_path == "cargo::commands::cargo-build"


def test_metadata_prefers_breadcrumbs_for_page_item_path():
    item_path = build_item_path(
        facts(
            file_path=Path("rust-docs/book/index.html"),
            crate=Crate.BOOK,
            breadcrumbs=("book", "Getting started"),
        )
    )

    assert item_path == "book::Getting started"


def test_metadata_detects_page_and_rustdoc_item_types():
    assert detect_item_type(facts(crate=Crate.REFERENCE)) == ItemType.PAGE
    assert detect_item_type(facts()) == ItemType.STRUCT
    assert detect_item_type(
        facts(
            file_path=Path("rust-docs/std/keyword.async.html"),
            title="std::keyword::async",
            text="async keyword",
        )
    ) == ItemType.KEYWORD
