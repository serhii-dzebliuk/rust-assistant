"""Chunk deduplication policy functions."""

from __future__ import annotations

from dataclasses import replace
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document


def _normalized_text_hash(text: str) -> str:
    """
    Create a stable normalized-text hash key.

    Args:
        text: Chunk text to normalize.

    Returns:
        SHA-256 hash for deduplication.
    """

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return Chunk.compute_text_hash(normalized)


def _chunk_selection_key(chunk: Chunk) -> tuple[int, int, int, str]:
    """
    Build a deterministic canonical-selection key for duplicate chunks.

    Args:
        chunk: Candidate duplicate chunk.

    Returns:
        Sort key preferring shorter, more canonical paths and earlier chunks.
    """

    source_path = chunk.source_path
    section_path = "::".join(chunk.section_path)
    return (
        source_path.count("/"),
        len(source_path),
        len(section_path),
        f"{source_path}:{chunk.chunk_index}",
    )


def _restore_orphan_document_chunks(
    chunks: list[Chunk],
    keep_indexes: set[int],
    indexes_by_source_path: dict[str, list[int]],
    documents: Sequence[Document],
) -> set[int]:
    """
    Restore one chunk for each document that lost all chunks after deduplication.

    Args:
        chunks: Original chunks before deduplication.
        keep_indexes: Currently kept chunk indexes.
        indexes_by_source_path: Original indexes grouped by source path.
        documents: Documents that must remain reachable.

    Returns:
        Updated keep-index set.
    """

    restored_indexes = set(keep_indexes)
    kept_source_paths = {chunks[index].source_path for index in restored_indexes}
    for document in documents:
        if document.source_path in kept_source_paths:
            continue

        candidate_indexes = indexes_by_source_path.get(document.source_path, [])
        if not candidate_indexes:
            continue

        best_index = min(
            candidate_indexes,
            key=lambda index: (
                chunks[index].chunk_index,
                len(chunks[index].text),
                chunks[index].start_offset,
                chunks[index].source_path,
            ),
        )
        restored_indexes.add(best_index)
        kept_source_paths.add(document.source_path)

    return restored_indexes


def _reindex_chunks_by_document(chunks: list[Chunk]) -> list[Chunk]:
    """
    Rebuild chunk indexes for each document after deduplication gaps.

    Args:
        chunks: Deduplicated chunks in pipeline output order.

    Returns:
        Chunks with contiguous per-document indexes while preserving the
        pipeline's document/chunk ordering.
    """

    positions_by_document: dict[str, list[int]] = defaultdict(list)
    for position, chunk in enumerate(chunks):
        positions_by_document[chunk.source_path].append(position)

    updated_by_position: dict[int, Chunk] = {}
    for positions in positions_by_document.values():
        ordered_positions = sorted(
            positions,
            key=lambda position: (
                chunks[position].start_offset,
                chunks[position].end_offset,
                chunks[position].chunk_index,
            ),
        )
        for chunk_index, position in enumerate(ordered_positions):
            chunk = chunks[position]
            updated_by_position[position] = replace(chunk, chunk_index=chunk_index)

    return [updated_by_position[position] for position in range(len(chunks))]


def deduplicate_chunks(
    chunks: list[Chunk],
    documents: Optional[Sequence[Document]] = None,
) -> list[Chunk]:
    """
    Deduplicate chunks while preserving original order of kept entries.

    Args:
        chunks: Generated chunks before chunk-level deduplication.
        documents: Optional document set that must remain reachable by at
            least one chunk after deduplication.

    Returns:
        Deduplicated chunk list.
    """

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    indexes_by_source_path: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        key = (chunk.crate.value, _normalized_text_hash(chunk.text))
        groups[key].append(index)
        indexes_by_source_path[chunk.source_path].append(index)

    keep_indexes: set[int] = set()
    for indexes in groups.values():
        if len(indexes) == 1:
            keep_indexes.add(indexes[0])
            continue

        best_index = min(indexes, key=lambda current: _chunk_selection_key(chunks[current]))
        keep_indexes.add(best_index)

    if documents is not None:
        keep_indexes = _restore_orphan_document_chunks(
            chunks,
            keep_indexes,
            indexes_by_source_path,
            documents,
        )

    deduplicated_chunks = [
        chunk for index, chunk in enumerate(chunks) if index in keep_indexes
    ]
    return _reindex_chunks_by_document(deduplicated_chunks)
