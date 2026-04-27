"""Filtering helpers for low-value chunk fragments."""

from __future__ import annotations

import re

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.enums import Crate


_LOW_VALUE_TINY_TEXTS = {
    "added",
    "changed",
    "fixed",
    "removed",
    "nightly only",
    "documentation",
    "internal",
}
_TINY_CHUNK_CHARS = 80


def filter_low_value_chunks(
    chunks: list[Chunk],
    *,
    min_chunk_chars: int,
) -> list[Chunk]:
    """
    Drop low-value chunk fragments when adjacent content carries more value.

    Args:
        chunks: Candidate chunks in document order.
        min_chunk_chars: Minimum chunk size targeted by safe merges.

    Returns:
        Filtered chunk list.
    """

    filtered_chunks: list[Chunk] = []
    for index, chunk in enumerate(chunks):
        if _should_drop_chunk(chunk, min_chunk_chars=min_chunk_chars):
            continue
        if _is_heading_only_chunk(chunk) and _has_same_section_neighbor(chunks, index):
            continue
        filtered_chunks.append(chunk)
    return filtered_chunks


def _should_drop_chunk(
    chunk: Chunk,
    *,
    min_chunk_chars: int,
) -> bool:
    section_path = chunk.section_path
    normalized_text = re.sub(r"\s+", " ", chunk.text.strip().lower())
    if (
        len(chunk.text) < _TINY_CHUNK_CHARS
        and normalized_text in _LOW_VALUE_TINY_TEXTS
        and "```" not in chunk.text
        and not _contains_api_signature(chunk.text)
    ):
        return True

    if (
        chunk.crate == Crate.CARGO
        and chunk.source_path == "cargo/CHANGELOG.html"
        and len(section_path) == 2
        and len(chunk.text) < min_chunk_chars
    ):
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        if len(lines) <= 2:
            return True

    return False


def _is_heading_only_chunk(chunk: Chunk) -> bool:
    stripped = chunk.text.strip()
    if not stripped or "```" in chunk.text:
        return False
    section_path = chunk.section_path
    if not section_path:
        return False
    return stripped == section_path[-1].strip()


def _has_same_section_neighbor(chunks: list[Chunk], index: int) -> bool:
    current = chunks[index]
    current_path = current.section_path
    for neighbor_index in (index - 1, index + 1):
        if 0 <= neighbor_index < len(chunks):
            neighbor = chunks[neighbor_index]
            if neighbor.section_path == current_path:
                return True
    return False


def _contains_api_signature(text: str) -> bool:
    normalized = text.strip()
    return bool(
        re.search(
            r"\b(fn|impl|trait|struct|enum|type|const|pub|unsafe)\b",
            normalized,
        )
    )
