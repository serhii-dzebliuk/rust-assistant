from dataclasses import replace

import pytest

from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate
from rust_assistant.domain.policies.document_cleaning import clean_documents
from rust_assistant.domain.policies.document_deduplication import (
    deduplicate_documents,
)
from rust_assistant.domain.value_objects.structured_blocks import (
    BlockType,
    StructuredBlock,
)

pytestmark = pytest.mark.unit


def _make_doc(source_path: str, text: str, crate: Crate = Crate.STD) -> Document:
    title = source_path.split("/")[-1].replace(".html", "")
    return Document(
        source_path=source_path,
        title=title,
        text=text,
        crate=crate,
        url=f"https://example.invalid/{source_path}",
        item_path=f"{crate.value}::{title}",
        structured_blocks=[],
    )


def test_clean_stage_normalizes_text_and_drops_short_docs():
    docs = [
        _make_doc(
            "std/intrinsics/simd/fn.simd_fcos.html",
            "Function simd_ fcos\n\nÂ§ Examples\n\n`cdylib` s are supported.\n",
        ),
        _make_doc("std/os/index.html", "tiny text"),
    ]

    cleaned = clean_documents(docs)

    assert len(cleaned) == 1
    assert "simd_fcos" in cleaned[0].text
    assert "Examples" in cleaned[0].text
    assert "Â§ " not in cleaned[0].text
    assert "`cdylib`s" in cleaned[0].text


def test_clean_stage_does_not_glue_syntax_labels_with_type_names():
    docs = [
        _make_doc(
            "reference/types/tuple.html",
            "Tuple types\n\nSyntax TupleType -> ( Type, )\n",
            crate=Crate.REFERENCE,
        ),
    ]

    cleaned = clean_documents(docs)

    assert len(cleaned) == 1
    assert "Syntax TupleType" in cleaned[0].text
    assert "SyntaxTupleType" not in cleaned[0].text


def test_clean_stage_normalizes_structured_blocks_without_breaking_code():
    doc = _make_doc(
        "std/example.html",
        "Function simd_ fcos\n\nÃ‚Â§ Examples\n\nThis document is intentionally long enough.\n",
    )
    doc = replace(
        doc,
        structured_blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Ã‚Â§ Examples",
                html_tag="h2",
            ),
            StructuredBlock(
                block_type=BlockType.CODE_BLOCK,
                text="line1\r\nline2\r\n",
                html_tag="pre",
                code_language="rust",
            ),
        ],
    )

    cleaned = clean_documents([doc])

    assert len(cleaned) == 1
    assert cleaned[0].structured_blocks[0].text == "Examples"
    assert cleaned[0].structured_blocks[1].text == "line1\nline2"


def test_clean_stage_rebuilds_text_from_structured_blocks_for_reference_docs():
    doc = _make_doc(
        "reference/procedural-macros.html",
        "Procedural Macros\n\n```toml\nproc-macro = true\n```\n",
        crate=Crate.REFERENCE,
    )
    doc = replace(
        doc,
        structured_blocks=[
            StructuredBlock(
                block_type=BlockType.HEADING,
                text="Procedural Macros",
                html_tag="h1",
                section_path=["Procedural Macros"],
                anchor="procedural-macros",
                heading_level=1,
            ),
            StructuredBlock(
                block_type=BlockType.CODE_BLOCK,
                text="[lib]\nproc-macro = true",
                html_tag="pre",
                code_language="toml",
                section_path=["Procedural Macros"],
                anchor="procedural-macros",
            ),
        ],
    )

    cleaned = clean_documents([doc])

    assert len(cleaned) == 1
    assert "[lib]" in cleaned[0].text
    assert cleaned[0].text == "Procedural Macros\n\n```toml\n[lib]\nproc-macro = true\n```"


def test_clean_stage_normalizes_punctuation_spacing_in_structured_blocks():
    doc = _make_doc(
        "std/index.html",
        "The standard library is portable software .",
    )
    doc = replace(
        doc,
        structured_blocks=[
            StructuredBlock(
                block_type=BlockType.PARAGRAPH,
                text="The standard library is portable software .",
                html_tag="p",
                section_path=["std"],
                anchor="main-content",
            ),
        ],
    )

    cleaned = clean_documents([doc])

    assert len(cleaned) == 1
    assert cleaned[0].structured_blocks[0].text == "The standard library is portable software."
    assert cleaned[0].text == "The standard library is portable software."


def test_document_dedup_removes_exact_duplicates_and_keeps_canonical_path():
    docs = [
        _make_doc("std/collections/struct.BTreeMap.html", "same text\n\nwith spacing"),
        _make_doc("std/collections/btree_map/struct.BTreeMap.html", "same text with spacing"),
        _make_doc("std/alloc/struct.Layout.html", "different text"),
    ]

    deduped = deduplicate_documents(docs)

    paths = {doc.source_path for doc in deduped}
    assert len(deduped) == 2
    assert "std/collections/struct.BTreeMap.html" in paths
    assert "std/collections/btree_map/struct.BTreeMap.html" not in paths
    assert "std/alloc/struct.Layout.html" in paths
