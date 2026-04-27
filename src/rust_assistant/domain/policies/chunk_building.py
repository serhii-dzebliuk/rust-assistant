"""Shared helpers for building chunks from document spans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document


def build_chunk(
    document: Document,
    *,
    start_offset: int,
    end_offset: int,
    chunk_index: int,
    section_path: Optional[Sequence[str]],
    anchor: Optional[str],
) -> Chunk:
    """
    Build one chunk from a document character span and section context.

    Args:
        document: Parent document.
        start_offset: Inclusive start offset in the document text.
        end_offset: Exclusive end offset in the document text.
        chunk_index: Chunk ordinal inside the document.
        section_path: Active section path for the span.
        anchor: Anchor associated with the chunk section.

    Returns:
        Chunk instance with metadata copied from the parent document.
    """

    return Chunk(
        source_path=document.source_path,
        chunk_index=chunk_index,
        text=document.text[start_offset:end_offset],
        crate=document.crate,
        start_offset=start_offset,
        end_offset=end_offset,
        section_path=tuple(section_path or ()),
        section_anchor=anchor,
        item_path=document.item_path,
        item_type=document.item_type,
        rust_version=document.rust_version,
        url=document.url,
    )


def build_fallback_chunk(document: Document) -> Chunk:
    """
    Build one full-document fallback chunk when structured blocks are unavailable.

    Args:
        document: Source document.

    Returns:
        One chunk covering the whole document text.
    """

    return build_chunk(
        document,
        start_offset=0,
        end_offset=len(document.text),
        chunk_index=0,
        section_path=(document.title,),
        anchor=None,
    )
