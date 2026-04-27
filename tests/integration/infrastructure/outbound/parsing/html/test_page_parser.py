from pathlib import Path

import pytest

from rust_assistant.domain.value_objects.structured_blocks import BlockType

pytestmark = pytest.mark.integration


def test_reference_destructors_keeps_code_comments_and_removes_rule_anchors(
    page_parser,
    raw_data_dir: Path,
):
    file_path = raw_data_dir / "reference/destructors.html"

    doc = page_parser.parse_file(file_path)
    assert doc is not None

    # Must keep code comment lines from examples.
    assert "// No destructor run on assignment." in doc.text
    # Must not leak reference anchor boilerplate from .rule blocks.
    assert "[destructors" not in doc.text
    # Known noisy marker in reference grammar pages.
    assert "Show Railroad" not in doc.text


def test_reference_destructors_keeps_sublist_content(page_parser, raw_data_dir: Path):
    file_path = raw_data_dir / "reference/destructors.html"

    doc = page_parser.parse_file(file_path)
    assert doc is not None

    assert "In the case of a block expression" in doc.text
    assert "scope for the block and the expression are the same scope." in doc.text


def test_book_hello_world_preserves_inline_code_and_avoids_nested_pre_duplicates(
    page_parser,
    raw_data_dir: Path,
):
    file_path = raw_data_dir / "book/ch01-02-hello-world.html"

    doc = page_parser.parse_file(file_path)
    assert doc is not None

    assert "`println!` calls a Rust macro." in doc.text
    assert doc.text.count('```rust\nfn main() {\n    println!("Hello, world!");\n}\n```') == 1


def test_book_hello_cargo_uses_console_fences_and_preserves_inline_commands(
    page_parser,
    raw_data_dir: Path,
):
    file_path = raw_data_dir / "book/ch01-03-hello-cargo.html"

    doc = page_parser.parse_file(file_path)
    assert doc is not None

    assert "```console\n$ cargo --version\n```" in doc.text
    assert "```console\n$ cargo run" in doc.text
    assert "We can create a project using `cargo new`." in doc.text
    assert "We can build and run a project in one step using `cargo run`." in doc.text


def test_book_parse_preserves_structured_blocks_for_headings_and_code(
    page_parser, raw_data_dir: Path
):
    doc = page_parser.parse_file(raw_data_dir / "book/ch01-03-hello-cargo.html")

    assert doc is not None
    assert doc.structured_blocks
    assert doc.structured_blocks[0].block_type == BlockType.HEADING
    assert doc.structured_blocks[0].text == "Hello, Cargo!"

    console_blocks = [
        block
        for block in doc.structured_blocks
        if block.block_type == BlockType.CODE_BLOCK and block.code_language == "console"
    ]
    assert console_blocks
    assert any(
        "Building and Running a Cargo Project" in block.section_path for block in console_blocks
    )


def test_book_metadata_uses_page_path_and_sets_url_and_version(page_parser, raw_data_dir: Path):
    file_path = raw_data_dir / "book/ch01-03-hello-cargo.html"

    doc = page_parser.parse_file(file_path)
    assert doc is not None

    assert doc.source_path == "book/ch01-03-hello-cargo.html"
    assert doc.item_path == "book::ch01-03-hello-cargo"
    assert doc.item_type == "page"
    assert doc.rust_version == "1.85.0"
    assert doc.url == "https://doc.rust-lang.org/book/ch01-03-hello-cargo.html"


def test_rustdoc_metadata_uses_canonical_item_path_and_version(page_parser, raw_data_dir: Path):
    file_path = raw_data_dir / "std/alloc/struct.Layout.html"

    doc = page_parser.parse_file(file_path)
    assert doc is not None

    assert doc.source_path == "std/alloc/struct.Layout.html"
    assert doc.item_path == "std::alloc::Layout"
    assert doc.item_type == "struct"
    assert doc.rust_version == "1.91.1"
    assert doc.url == "https://doc.rust-lang.org/std/alloc/struct.Layout.html"


def test_rustdoc_detects_trait_module_and_keyword_item_types(page_parser, raw_data_dir: Path):

    trait_doc = page_parser.parse_file(raw_data_dir / "std/alloc/trait.Allocator.html")
    assert trait_doc is not None
    assert trait_doc.item_type == "trait"

    module_doc = page_parser.parse_file(raw_data_dir / "std/alloc/index.html")
    assert module_doc is not None
    assert module_doc.item_type == "module"

    keyword_doc = page_parser.parse_file(raw_data_dir / "std/keyword.async.html")
    assert keyword_doc is not None
    assert keyword_doc.item_type == "keyword"

    primitive_doc = page_parser.parse_file(raw_data_dir / "std/primitive.unit.html")
    assert primitive_doc is not None
    assert primitive_doc.item_type == "primitive"


def test_rustdoc_uses_rust_fences_for_rustdoc_code_blocks(page_parser, raw_data_dir: Path):
    doc = page_parser.parse_file(raw_data_dir / "std/alloc/struct.Layout.html")

    assert doc is not None
    assert "```rust" in doc.text
    assert "pub struct Layout" in doc.text


def test_rustdoc_parse_preserves_section_paths_and_anchors(page_parser, raw_data_dir: Path):
    doc = page_parser.parse_file(raw_data_dir / "std/vec/struct.Vec.html")

    assert doc is not None
    example_heading = next(
        (
            block
            for block in doc.structured_blocks
            if block.block_type == BlockType.HEADING and block.text == "Examples"
        ),
        None,
    )
    assert example_heading is not None
    assert example_heading.heading_level is not None
    assert example_heading.anchor is not None

    example_code_blocks = [
        block
        for block in doc.structured_blocks
        if block.block_type == BlockType.CODE_BLOCK and "Examples" in block.section_path
    ]
    assert example_code_blocks


def test_rustdoc_code_blocks_do_not_insert_spurious_line_breaks(page_parser, raw_data_dir: Path):
    doc = page_parser.parse_file(raw_data_dir / "std/alloc/trait.Allocator.html")

    assert doc is not None
    assert "fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;" in doc.text
    assert "unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);" in doc.text


def test_rustdoc_module_index_keeps_item_table_entries(page_parser, raw_data_dir: Path):
    doc = page_parser.parse_file(raw_data_dir / "std/intrinsics/fallback/index.html")

    assert doc is not None
    assert "std::intrinsics::fallback" == doc.item_path
    assert doc.item_type == "module"
    assert "Carrying MulAdd" in doc.text
    assert "Disjoint BitOr" in doc.text


def test_cargo_metadata_uses_page_path_sets_url_and_null_item_type(page_parser, raw_data_dir: Path):
    file_path = raw_data_dir / "cargo/commands/cargo-build.html"

    doc = page_parser.parse_file(file_path)
    assert doc is not None

    assert doc.item_path == "cargo::commands::cargo-build"
    assert doc.item_type == "page"
    assert doc.rust_version == "1.91"
    assert doc.url == "https://doc.rust-lang.org/cargo/commands/cargo-build.html"


def test_reference_metadata_uses_page_path_sets_url_and_keeps_version_null(
    page_parser, raw_data_dir: Path
):
    file_path = raw_data_dir / "reference/destructors.html"

    doc = page_parser.parse_file(file_path)
    assert doc is not None

    assert doc.item_path == "reference::destructors"
    assert doc.item_type == "page"
    assert doc.rust_version is None
    assert doc.url == "https://doc.rust-lang.org/reference/destructors.html"


def test_reference_parse_preserves_structured_blocks(page_parser, raw_data_dir: Path):
    doc = page_parser.parse_file(raw_data_dir / "reference/types/tuple.html")

    assert doc is not None
    assert doc.structured_blocks[0].block_type == BlockType.HEADING
    assert doc.structured_blocks[0].text == "Tuple types"
    assert any(block.block_type == BlockType.LIST_ITEM for block in doc.structured_blocks)
