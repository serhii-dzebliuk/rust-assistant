"""Merging helpers for adjacent chunk fragments."""

from __future__ import annotations

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate
from rust_assistant.domain.policies.chunk_building import build_chunk


def merge_small_chunks(
    document: Document,
    chunks: list[Chunk],
    *,
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    """
    Conservatively merge undersized adjacent chunks when they share context.

    Args:
        document: Parent document.
        chunks: Chunks produced from section splitting.
        max_chunk_chars: Maximum allowed merged chunk size.
        min_chunk_chars: Minimum chunk size targeted by safe merges.

    Returns:
        Chunk list with obvious tiny parent-intro chunks merged forward.
    """

    if len(chunks) < 2:
        return chunks

    merged_chunks: list[Chunk] = []
    current_chunk = chunks[0]
    for next_chunk in chunks[1:]:
        if _should_merge_chunks(
            document,
            current_chunk,
            next_chunk,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        ):
            current_chunk = _merge_two_chunks(document, current_chunk, next_chunk)
            continue
        merged_chunks.append(current_chunk)
        current_chunk = next_chunk

    merged_chunks.append(current_chunk)
    return merged_chunks


def reindex_chunks(document: Document, chunks: list[Chunk]) -> list[Chunk]:
    """
    Rebuild chunk ids and chunk indexes after merge and split passes.

    Args:
        document: Parent document.
        chunks: Final chunk list in document order.

    Returns:
        Reindexed chunk list.
    """

    reindexed_chunks: list[Chunk] = []
    for chunk_index, chunk in enumerate(chunks):
        reindexed_chunks.append(
            build_chunk(
                document,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                chunk_index=chunk_index,
                section_path=_chunk_section_path(chunk, document.title),
                anchor=chunk.section_anchor,
            )
        )
    return reindexed_chunks


def _should_merge_chunks(
    document: Document,
    left: Chunk,
    right: Chunk,
    *,
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> bool:
    if len(left.text) >= min_chunk_chars:
        return False

    if not _chunks_are_contiguous(document, left, right):
        return False

    if "```" in left.text and left.section_path != right.section_path:
        return False

    merged_length = right.end_offset - left.start_offset
    if merged_length > max_chunk_chars:
        return False

    left_path = left.section_path
    right_path = right.section_path
    if not left_path or not right_path:
        return False

    if left_path == right_path or _is_prefix_path(left_path, right_path):
        return True

    crate = left.crate
    if crate in {Crate.BOOK, Crate.CARGO, Crate.REFERENCE}:
        if (
            len(left.text) < min_chunk_chars
            and len(right.text) < min_chunk_chars
            and _share_parent_path(left_path, right_path)
        ):
            return True

    return False


def _chunks_are_contiguous(document: Document, left: Chunk, right: Chunk) -> bool:
    if left.end_offset > right.start_offset:
        return False
    gap = document.text[left.end_offset : right.start_offset]
    return not gap.strip()


def _merge_two_chunks(document: Document, left: Chunk, right: Chunk) -> Chunk:
    left_path = left.section_path
    right_path = right.section_path
    if _is_prefix_path(left_path, right_path):
        section_path = left.section_path
        anchor = left.section_anchor or right.section_anchor
    elif _share_parent_path(left_path, right_path):
        section_path = left_path[:-1]
        anchor = left.section_anchor or right.section_anchor
    else:
        section_path = left.section_path
        anchor = left.section_anchor or right.section_anchor

    return build_chunk(
        document,
        start_offset=min(left.start_offset, right.start_offset),
        end_offset=max(left.end_offset, right.end_offset),
        chunk_index=left.chunk_index,
        section_path=section_path or _chunk_section_path(left, document.title),
        anchor=anchor,
    )


def _chunk_section_path(chunk: Chunk, document_title: str) -> tuple[str, ...]:
    if chunk.section_path:
        return chunk.section_path
    return (document_title,)


def _is_prefix_path(prefix: tuple[str, ...], candidate: tuple[str, ...]) -> bool:
    if len(prefix) >= len(candidate):
        return False
    return candidate[: len(prefix)] == prefix


def _share_parent_path(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if len(left) != len(right):
        return False
    if len(left) < 3:
        return False
    return left[:-1] == right[:-1] and left[-1] != right[-1]
