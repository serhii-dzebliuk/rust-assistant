"""
Exact deduplication for retrieval chunks.

This module implements ingest stage 1.7.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from collections import defaultdict
from typing import Optional

from rust_assistant.ingest.entities import Chunk, Document

logger = logging.getLogger(__name__)


class ChunkDeduplicator:
    """
    Perform exact deduplication on retrieval chunks.

    Duplicate groups are computed from crate plus normalized chunk text.
    """

    def _normalized_text_hash(self, text: str) -> str:
        """
        Create a stable normalized-text hash key.

        Args:
            text: Chunk text to normalize.

        Returns:
            SHA-256 hash for deduplication.
        """
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return Chunk.compute_text_hash(normalized)

    def _selection_key(self, chunk: Chunk) -> tuple[int, int, int, str]:
        """
        Build a deterministic canonical-selection key for duplicates.

        Args:
            chunk: Candidate duplicate chunk.

        Returns:
            Sort key preferring shorter, more canonical paths and earlier chunks.
        """
        source = chunk.metadata.doc_source_path.replace("\\", "/")
        section_path = "::".join(chunk.metadata.section_path or [])
        return (
            source.count("/"),
            len(source),
            len(section_path),
            f"{source}:{chunk.metadata.chunk_index}",
        )

    def deduplicate(
        self,
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
            key = (
                chunk.metadata.crate.value,
                self._normalized_text_hash(chunk.text),
            )
            groups[key].append(index)
            indexes_by_source_path[chunk.metadata.doc_source_path].append(index)

        keep_indexes: set[int] = set()
        duplicate_groups = 0
        duplicate_chunks = 0
        for indexes in groups.values():
            if len(indexes) == 1:
                keep_indexes.add(indexes[0])
                continue

            duplicate_groups += 1
            duplicate_chunks += len(indexes) - 1
            best_index = min(indexes, key=lambda idx: self._selection_key(chunks[idx]))
            keep_indexes.add(best_index)

        restored_orphan_chunks = 0
        if documents is not None:
            kept_source_paths = {chunks[index].metadata.doc_source_path for index in keep_indexes}
            for document in documents:
                if document.source_path in kept_source_paths:
                    continue
                candidate_indexes = indexes_by_source_path.get(document.source_path, [])
                if not candidate_indexes:
                    continue
                best_index = min(
                    candidate_indexes,
                    key=lambda idx: (
                        chunks[idx].metadata.chunk_index,
                        len(chunks[idx].text),
                        chunks[idx].chunk_id,
                    ),
                )
                keep_indexes.add(best_index)
                kept_source_paths.add(document.source_path)
                restored_orphan_chunks += 1

        deduped = [chunk for index, chunk in enumerate(chunks) if index in keep_indexes]
        deduped = self._reindex_by_document(deduped)
        if duplicate_groups:
            logger.info(
                "Chunk dedup removed %s chunks across %s duplicate groups",
                duplicate_chunks,
                duplicate_groups,
            )
        if restored_orphan_chunks:
            logger.info(
                "Chunk dedup restored %s chunks to keep documents reachable",
                restored_orphan_chunks,
            )
        return deduped

    def _reindex_by_document(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Rebuild chunk indexes for each document after deduplication gaps.

        Args:
            chunks: Deduplicated chunks in pipeline output order.

        Returns:
            Chunks with contiguous per-document indexes while preserving the
            pipeline's document/chunk ordering.
        """
        positions_by_document: dict[tuple[str, str], list[int]] = defaultdict(list)
        for position, chunk in enumerate(chunks):
            key = (chunk.doc_id, chunk.metadata.doc_source_path)
            positions_by_document[key].append(position)

        updated_by_position: dict[int, Chunk] = {}
        for positions in positions_by_document.values():
            ordered_positions = sorted(
                positions,
                key=lambda position: (
                    chunks[position].metadata.start_char,
                    chunks[position].metadata.end_char,
                    chunks[position].metadata.chunk_index,
                ),
            )
            for chunk_index, position in enumerate(ordered_positions):
                chunk = chunks[position]
                metadata = chunk.metadata.model_copy(update={"chunk_index": chunk_index})
                updated_by_position[position] = Chunk(
                    chunk_id=Chunk.generate_id(chunk.doc_id, chunk_index),
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    metadata=metadata,
                )

        return [updated_by_position[position] for position in range(len(chunks))]


def deduplicate_chunks(
    chunks: list[Chunk],
    documents: Optional[Sequence[Document]] = None,
) -> list[Chunk]:
    """
    Deduplicate chunks in memory.

    Args:
        chunks: Chunks produced by the chunking stage.
        documents: Optional document set that must remain reachable.

    Returns:
        Deduplicated chunk list.
    """
    logger.info("Deduplicating %s chunks...", len(chunks))
    deduplicator = ChunkDeduplicator()
    deduped = deduplicator.deduplicate(chunks, documents=documents)
    logger.info("Chunk dedup complete: kept %s chunks", len(deduped))

    return deduped
