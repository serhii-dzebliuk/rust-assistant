import pytest

from rust_assistant.ingest.chunk import DocumentChunker, chunk_documents
from rust_assistant.ingest.chunk_dedup import deduplicate_chunks
from rust_assistant.ingest.parsing.core import blocks_to_text
from rust_assistant.ingest.entities import (
    BlockType,
    Chunk,
    Document,
    DocumentMetadata,
    StructuredBlock,
)
from rust_assistant.schemas.enums import Crate

pytestmark = pytest.mark.unit


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
    assert [chunk.metadata.section for chunk in chunks] == ["Struct Vec", "Panics"]
    assert chunks[0].metadata.anchor == "main-content"
    assert chunks[0].metadata.section_path == ["Struct Vec"]
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
        chunk.text == doc.text[chunk.metadata.start_char : chunk.metadata.end_char]
        for chunk in chunks
    )
    assert all(not chunk.text.startswith("Hello, Cargo!") for chunk in chunks[1:])


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
        chunk.text == doc.text[chunk.metadata.start_char : chunk.metadata.end_char]
        for chunk in chunks
    )
    reconstructed = "".join(
        doc.text[chunk.metadata.start_char : chunk.metadata.end_char] for chunk in chunks
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
    assert any(
        chunk.text.lstrip().startswith('"key_') or chunk.text.lstrip().startswith("{")
        for chunk in chunks[1:]
    )
    assert all(
        chunk.text == doc.text[chunk.metadata.start_char : chunk.metadata.end_char]
        for chunk in chunks
    )


def test_chunker_aligns_block_spans_to_actual_document_text():
    doc = _make_doc(
        source_path="std/primitive.array.html",
        title="Primitive Type array",
        crate=Crate.STD,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Primitive Type array",
                html_tag="h1",
                heading_level=1,
                anchor="main-content",
                section_path=["Primitive Type array"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="The initializer must either be:",
                html_tag="p",
                anchor="main-content",
                section_path=["Primitive Type array"],
            ),
            StructuredBlock(
                block_type=BlockType.LIST_ITEM,
                text="A value of a type implementing the `Copy` trait",
                html_tag="li",
                list_depth=0,
                anchor="main-content",
                section_path=["Primitive Type array"],
            ),
        ],
    )
    doc = doc.model_copy(update={"text": doc.text.replace("\n\n- A value", "\n\n  - A value")})

    rendered_blocks, block_spans = DocumentChunker()._build_block_spans(
        doc.structured_blocks,
        doc.text,
    )
    list_item_index = 2
    start_char, end_char = block_spans[list_item_index]

    assert start_char == doc.text.index("- A value")
    assert doc.text[start_char:end_char] == rendered_blocks[list_item_index]


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


def test_chunk_dedup_restores_one_chunk_for_each_document():
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
        title="Chapter 1 copy",
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
    deduped = deduplicate_chunks(chunks, documents=[doc_a, doc_b])

    assert len(deduped) == 2
    assert {chunk.metadata.doc_source_path for chunk in deduped} == {
        "book/ch01.html",
        "book/ch01-copy.html",
    }


def test_chunk_dedup_reindexes_chunks_per_document_after_removing_duplicates():
    doc = _make_doc(
        source_path="std/example.html",
        title="std::example",
        crate=Crate.STD,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Example",
                html_tag="h1",
                heading_level=1,
                anchor="example",
                section_path=["Example"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="Duplicate text.",
                html_tag="p",
                anchor="example",
                section_path=["Example"],
            ),
        ],
    )
    first = chunk_documents([doc])[0]
    duplicate = Chunk(
        chunk_id=Chunk.generate_id(doc.doc_id, 5),
        doc_id=doc.doc_id,
        text=first.text,
        metadata=first.metadata.model_copy(
            update={
                "chunk_index": 5,
                "start_char": 100,
                "end_char": 115,
            }
        ),
    )
    unique = Chunk(
        chunk_id=Chunk.generate_id(doc.doc_id, 9),
        doc_id=doc.doc_id,
        text="Unique later text.",
        metadata=first.metadata.model_copy(
            update={
                "chunk_index": 9,
                "start_char": 200,
                "end_char": 218,
            }
        ),
    )

    deduped = deduplicate_chunks([first, duplicate, unique])

    assert [chunk.metadata.chunk_index for chunk in deduped] == [0, 1]
    assert [chunk.chunk_id for chunk in deduped] == [
        Chunk.generate_id(doc.doc_id, 0),
        Chunk.generate_id(doc.doc_id, 1),
    ]


def test_chunker_splits_oversized_non_code_sections_on_safe_boundaries():
    long_impl_text = "\n\n".join(
        f"fn generated_method_{index} (&self) -> usize" for index in range(30)
    )
    doc = _make_doc(
        source_path="std/str/struct.Matches.html",
        title="std::str::Matches",
        crate=Crate.STD,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Struct Matches",
                html_tag="h1",
                heading_level=1,
                anchor="main-content",
                section_path=["Struct Matches"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="Created with the method `matches`.",
                html_tag="p",
                anchor="main-content",
                section_path=["Struct Matches"],
            ),
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Trait Implementations",
                html_tag="h2",
                heading_level=2,
                anchor="trait-implementations",
                section_path=["Struct Matches", "Trait Implementations"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text=long_impl_text,
                html_tag="p",
                anchor="trait-implementations",
                section_path=["Struct Matches", "Trait Implementations"],
            ),
        ],
    )

    chunks = DocumentChunker(max_chunk_chars=260).chunk_document(doc)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 260 for chunk in chunks)
    assert all(
        chunk.text == doc.text[chunk.metadata.start_char : chunk.metadata.end_char]
        for chunk in chunks
    )


def test_text_boundary_splitter_avoids_tiny_tail_fragments():
    text = "This sentence has several useful words. " * 6

    groups = DocumentChunker()._split_text_by_boundaries(text, 170)

    assert len(groups) == 2
    assert all(len(group_text) <= 170 for group_text, _ in groups)
    assert len(groups[-1][0].strip()) >= 80
    assert "".join(group_text for group_text, _ in groups) == text


def test_chunker_does_not_emit_heading_only_chunk_before_oversized_code_block():
    code = "\n".join(f"let value_{index} = {index};" for index in range(30))
    doc = _make_doc(
        source_path="std/ops/enum.ControlFlow.html",
        title="std::ops::ControlFlow",
        crate=Crate.STD,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Examples",
                html_tag="h2",
                heading_level=2,
                anchor="examples",
                section_path=["std::ops::ControlFlow", "Examples"],
            ),
            StructuredBlock(
                block_type=BlockType.CODE_BLOCK,
                text=code,
                html_tag="pre",
                code_language="rust",
                anchor="examples",
                section_path=["std::ops::ControlFlow", "Examples"],
            ),
        ],
    )

    chunks = DocumentChunker(max_chunk_chars=180).chunk_document(doc)

    assert len(chunks) > 1
    assert all(chunk.text != "Examples\n\n" for chunk in chunks)
    assert all(len(chunk.text) <= 180 for chunk in chunks)
    assert chunks[0].text.startswith("Examples")
    assert "```rust" in chunks[0].text


def test_chunker_refuses_to_split_single_oversized_code_line():
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
                text='let value = "' + ("x" * 220) + '";',
                html_tag="pre",
                code_language="rust",
                anchor="examples",
                section_path=["std::example", "Examples"],
            ),
        ],
    )

    with pytest.raises(ValueError, match="Cannot safely split a code block line"):
        DocumentChunker(max_chunk_chars=120).chunk_document(doc)


def test_chunker_refuses_blind_split_without_text_boundary():
    doc = _make_doc(
        source_path="reference/long-token.html",
        title="Long token",
        crate=Crate.REFERENCE,
        blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Long token",
                html_tag="h1",
                heading_level=1,
                anchor="long-token",
                section_path=["Long token"],
            ),
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="x" * 240,
                html_tag="p",
                anchor="long-token",
                section_path=["Long token"],
            ),
        ],
    )

    with pytest.raises(ValueError, match="without a semantic boundary"):
        DocumentChunker(max_chunk_chars=120).chunk_document(doc)
