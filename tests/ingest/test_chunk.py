from rustrag.ingest.chunk import DocumentChunker, chunk_documents
from rustrag.ingest.chunk_dedup import deduplicate_chunks
from rustrag.ingest.parsing.core import blocks_to_text
from rustrag.core.models import (
    BlockType,
    Crate,
    Document,
    DocumentMetadata,
    StructuredBlock,
)


def _make_doc(
    source_path: str,
    title: str,
    crate: Crate,
    blocks: list[StructuredBlock],
) -> Document:
    text = blocks_to_text(blocks)
    metadata = DocumentMetadata(
        crate=crate,
        item_path=f"{crate.value}::{title.replace(' ', '_')}",
        raw_html_path=f"D:\\tmp\\{source_path.replace('/', '\\')}",
    )
    return Document(
        doc_id=Document.generate_id(source_path, title),
        title=title,
        source_path=source_path,
        text=text,
        structured_blocks=blocks,
        metadata=metadata,
    )


def test_rustdoc_chunker_splits_by_sections_and_keeps_section_metadata():
    doc = _make_doc(
        source_path="std/vec/struct.Vec.html",
        title="std::vec::Vec",
        crate=Crate.STD,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Struct Vec",
                html_tag="h1",
                heading_level=1,
                anchor="main-content",
                section_path=["Struct Vec"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="A contiguous growable array type.",
                html_tag="p",
                anchor="main-content",
                section_path=["Struct Vec"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Examples",
                html_tag="h2",
                heading_level=2,
                anchor="examples",
                section_path=["Struct Vec", "Examples"],
            ),
            StructuredBlock(
                block_type=BlockType.CODE_BLOCK,
                text="let mut vec = Vec::new();\nvec.push(1);",
                html_tag="pre",
                code_language="rust",
                anchor="examples",
                section_path=["Struct Vec", "Examples"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Panics",
                html_tag="h2",
                heading_level=2,
                anchor="panics",
                section_path=["Struct Vec", "Panics"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="Panics if the allocator cannot reserve memory.",
                html_tag="p",
                anchor="panics",
                section_path=["Struct Vec", "Panics"],
            ),
        ],
    )

    chunks = chunk_documents([doc])

    assert len(chunks) == 2
    assert [chunk.metadata.section for chunk in chunks] == ["Examples", "Panics"]
    assert chunks[0].metadata.anchor == "examples"
    assert chunks[0].metadata.section_path == ["Struct Vec", "Examples"]
    assert "A contiguous growable array type." in chunks[0].text
    assert "```rust" in chunks[0].text


def test_book_chunker_preserves_exact_text_spans_when_splitting_large_section():
    long_paragraph = " ".join(["Cargo builds projects quickly."] * 20)
    doc = _make_doc(
        source_path="book/ch01-03-hello-cargo.html",
        title="Hello, Cargo!",
        crate=Crate.BOOK,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Hello, Cargo!",
                html_tag="h1",
                heading_level=1,
                anchor="hello-cargo",
                section_path=["Hello, Cargo!"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text=long_paragraph,
                html_tag="p",
                anchor="hello-cargo",
                section_path=["Hello, Cargo!"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text=long_paragraph,
                html_tag="p",
                anchor="hello-cargo",
                section_path=["Hello, Cargo!"],
            ),
        ],
    )

    chunks = DocumentChunker(max_chunk_chars=220).chunk_document(doc)

    assert len(chunks) > 1
    assert all(chunk.metadata.section == "Hello, Cargo!" for chunk in chunks)
    assert chunks[0].text.startswith("Hello, Cargo!")
    assert all(
        chunk.text == doc.text[chunk.metadata.start_char:chunk.metadata.end_char]
        for chunk in chunks
    )
    assert all(
        not chunk.text.startswith("Hello, Cargo!")
        for chunk in chunks[1:]
    )


def test_chunker_splits_large_code_blocks_into_exact_document_slices():
    huge_code_block = "\n".join(f"println!({index});" for index in range(40))
    doc = _make_doc(
        source_path="std/example.html",
        title="std::example",
        crate=Crate.STD,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Examples",
                html_tag="h2",
                heading_level=2,
                anchor="examples",
                section_path=["std::example", "Examples"],
            ),
            StructuredBlock(
                block_type=BlockType.CODE_BLOCK,
                text=huge_code_block,
                html_tag="pre",
                code_language="rust",
                anchor="examples",
                section_path=["std::example", "Examples"],
            ),
        ],
    )

    chunks = DocumentChunker(max_chunk_chars=120).chunk_document(doc)

    assert len(chunks) > 1
    assert chunks[0].text.startswith("Examples")
    assert chunks[0].metadata.start_char == 0
    assert chunks[-1].text.rstrip().endswith("```")
    assert all(
        chunk.text == doc.text[chunk.metadata.start_char:chunk.metadata.end_char]
        for chunk in chunks
    )
    reconstructed = "".join(
        doc.text[chunk.metadata.start_char:chunk.metadata.end_char]
        for chunk in chunks
    )
    assert reconstructed == doc.text


def test_chunker_splits_oversized_code_block_even_after_intro_paragraph():
    huge_json = "\n".join(f'  "key_{index}": "{index}",' for index in range(50))
    doc = _make_doc(
        source_path="cargo/commands/cargo-metadata.html",
        title="cargo metadata",
        crate=Crate.CARGO,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="JSON format",
                html_tag="h2",
                heading_level=2,
                anchor="json-format",
                section_path=["cargo metadata", "JSON format"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="Example output for the command.",
                html_tag="p",
                anchor="json-format",
                section_path=["cargo metadata", "JSON format"],
            ),
            StructuredBlock(
                block_type=BlockType.CODE_BLOCK,
                text="{\n" + huge_json + "\n}",
                html_tag="pre",
                code_language="json",
                anchor="json-format",
                section_path=["cargo metadata", "JSON format"],
            ),
        ],
    )

    chunks = DocumentChunker(max_chunk_chars=160).chunk_document(doc)

    assert len(chunks) > 2
    assert chunks[0].text.startswith("JSON format")
    assert any(chunk.text.lstrip().startswith('"key_') or chunk.text.lstrip().startswith('{') for chunk in chunks[1:])
    assert all(
        chunk.text == doc.text[chunk.metadata.start_char:chunk.metadata.end_char]
        for chunk in chunks
    )


def test_cargo_chunker_preserves_section_path_metadata():
    doc = _make_doc(
        source_path="cargo/commands/cargo-run.html",
        title="cargo run",
        crate=Crate.CARGO,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="cargo-run(1)",
                html_tag="h1",
                heading_level=1,
                anchor="cargo-run",
                section_path=["cargo-run(1)"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="SYNOPSIS",
                html_tag="h2",
                heading_level=2,
                anchor="synopsis",
                section_path=["cargo-run(1)", "SYNOPSIS"],
            ),
            StructuredBlock(
                block_type=BlockType.CODE_BLOCK,
                text="cargo run [options] [-- args]",
                html_tag="pre",
                code_language="text",
                anchor="synopsis",
                section_path=["cargo-run(1)", "SYNOPSIS"],
            ),
        ],
    )

    chunks = chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata.section == "SYNOPSIS"
    assert chunks[0].metadata.section_path == ["cargo-run(1)", "SYNOPSIS"]


def test_chunker_merges_tiny_sibling_sections_for_reference_docs():
    doc = _make_doc(
        source_path="reference/conditional-compilation.html",
        title="Conditional compilation",
        crate=Crate.REFERENCE,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Conditional compilation",
                html_tag="h1",
                heading_level=1,
                anchor="conditional-compilation",
                section_path=["Conditional compilation"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Set Configuration Options",
                html_tag="h2",
                heading_level=2,
                anchor="set-configuration-options",
                section_path=["Conditional compilation", "Set Configuration Options"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="`target_endian`",
                html_tag="h3",
                heading_level=3,
                anchor="target-endian",
                section_path=[
                    "Conditional compilation",
                    "Set Configuration Options",
                    "`target_endian`",
                ],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="Key-value option set once with either little or big.",
                html_tag="p",
                anchor="target-endian",
                section_path=[
                    "Conditional compilation",
                    "Set Configuration Options",
                    "`target_endian`",
                ],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="`target_pointer_width`",
                html_tag="h3",
                heading_level=3,
                anchor="target-pointer-width",
                section_path=[
                    "Conditional compilation",
                    "Set Configuration Options",
                    "`target_pointer_width`",
                ],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="Key-value option set once with the target pointer width in bits.",
                html_tag="p",
                anchor="target-pointer-width",
                section_path=[
                    "Conditional compilation",
                    "Set Configuration Options",
                    "`target_pointer_width`",
                ],
            ),
        ],
    )

    chunks = chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata.section == "Set Configuration Options"
    assert chunks[0].metadata.section_path == [
        "Conditional compilation",
        "Set Configuration Options",
    ]
    assert "`target_endian`" in chunks[0].text
    assert "`target_pointer_width`" in chunks[0].text


def test_chunker_drops_empty_cargo_changelog_release_stub():
    doc = _make_doc(
        source_path="cargo/CHANGELOG.html",
        title="Changelog",
        crate=Crate.CARGO,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Changelog",
                html_tag="h1",
                heading_level=1,
                anchor="changelog",
                section_path=["Changelog"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Cargo 1.91 (2025-10-30)",
                html_tag="h2",
                heading_level=2,
                anchor="cargo-191",
                section_path=["Changelog", "Cargo 1.91 (2025-10-30)"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="840b83a1…HEAD",
                html_tag="p",
                anchor="cargo-191",
                section_path=["Changelog", "Cargo 1.91 (2025-10-30)"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Cargo 1.90 (2025-09-18)",
                html_tag="h2",
                heading_level=2,
                anchor="cargo-190",
                section_path=["Changelog", "Cargo 1.90 (2025-09-18)"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Added",
                html_tag="h3",
                heading_level=3,
                anchor="added",
                section_path=["Changelog", "Cargo 1.90 (2025-09-18)", "Added"],
            ),
            StructuredBlock(
                block_type=BlockType.LIST_ITEM,
                text="Stabilize multi-package publishing.",
                html_tag="li",
                list_depth=0,
                anchor="added",
                section_path=["Changelog", "Cargo 1.90 (2025-09-18)", "Added"],
            ),
        ],
    )

    chunks = chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata.section_path == ["Changelog", "Cargo 1.90 (2025-09-18)", "Added"]
    assert "Cargo 1.91 (2025-10-30)" not in chunks[0].text


def test_chunk_dedup_removes_exact_duplicate_chunks_within_same_crate():
    doc_a = _make_doc(
        source_path="book/ch01.html",
        title="Chapter 1",
        crate=Crate.BOOK,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Chapter 1",
                html_tag="h1",
                heading_level=1,
                anchor="chapter-1",
                section_path=["Chapter 1"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="Install Rust with rustup.",
                html_tag="p",
                anchor="chapter-1",
                section_path=["Chapter 1"],
            ),
        ],
    )
    doc_b = _make_doc(
        source_path="book/ch01-copy.html",
        title="Chapter 1",
        crate=Crate.BOOK,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Chapter 1",
                html_tag="h1",
                heading_level=1,
                anchor="chapter-1-copy",
                section_path=["Chapter 1"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="Install Rust with rustup.",
                html_tag="p",
                anchor="chapter-1-copy",
                section_path=["Chapter 1"],
            ),
        ],
    )

    chunks = chunk_documents([doc_a, doc_b])
    deduped = deduplicate_chunks(chunks)

    assert len(chunks) == 2
    assert len(deduped) == 1
    assert deduped[0].metadata.doc_source_path == "book/ch01.html"
