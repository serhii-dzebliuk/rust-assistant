"""Section-building helpers for chunking policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate
from rust_assistant.domain.value_objects.structured_blocks import (
    BlockType,
    StructuredBlock,
)


_MAX_CHUNK_CHARS_BY_CRATE = {
    Crate.STD: 1350,
    Crate.CORE: 1350,
    Crate.ALLOC: 1350,
    Crate.PROC_MACRO: 1350,
    Crate.TEST: 1350,
    Crate.BOOK: 1500,
    Crate.CARGO: 1450,
    Crate.REFERENCE: 1300,
}


@dataclass(slots=True, frozen=True)
class DocumentSection:
    """
    Contiguous document block span representing one logical section.

    Attributes:
        block_indexes: Ordered indexes of blocks belonging to the section.
        section_path: Heading path active for the section.
        anchor: Section anchor used for chunk metadata.
    """

    block_indexes: list[int]
    section_path: tuple[str, ...]
    anchor: Optional[str]


def resolve_chunk_limit(crate: Crate, max_chunk_chars: int) -> int:
    """
    Resolve the effective chunk-size cap for a crate.

    Args:
        crate: Document crate identifier.
        max_chunk_chars: User-level hard cap.

    Returns:
        Effective chunk-size limit for the document.
    """

    crate_limit = _MAX_CHUNK_CHARS_BY_CRATE.get(crate, 1400)
    return min(crate_limit, max_chunk_chars)


def build_document_sections(document: Document) -> list[DocumentSection]:
    """
    Split a document into contiguous logical sections.

    Args:
        document: Cleaned document with structured blocks.

    Returns:
        Ordered list of document sections.
    """

    if not document.structured_blocks:
        return []

    sections: list[DocumentSection] = []
    current_indexes: list[int] = []

    for index, block in enumerate(document.structured_blocks):
        if _starts_new_section(block):
            if current_indexes:
                sections.append(_build_document_section(document, current_indexes))
            current_indexes = [index]
            continue

        if not current_indexes:
            current_indexes = [index]
        else:
            current_indexes.append(index)

    if current_indexes:
        sections.append(_build_document_section(document, current_indexes))

    meaningful_sections = [
        section
        for section in sections
        if any(
            document.structured_blocks[index].block_type != BlockType.HEADING
            for index in section.block_indexes
        )
    ]
    return meaningful_sections or sections


def _starts_new_section(block: StructuredBlock) -> bool:
    return block.block_type == BlockType.HEADING


def _build_document_section(
    document: Document,
    block_indexes: list[int],
) -> DocumentSection:
    lead_block = document.structured_blocks[block_indexes[0]]
    section_path = lead_block.section_path
    if not section_path:
        section_path = (document.title,)

    anchor = lead_block.anchor
    if anchor is None:
        for index in block_indexes:
            candidate = document.structured_blocks[index].anchor
            if candidate is not None:
                anchor = candidate
                break

    return DocumentSection(
        block_indexes=block_indexes,
        section_path=section_path,
        anchor=anchor,
    )
