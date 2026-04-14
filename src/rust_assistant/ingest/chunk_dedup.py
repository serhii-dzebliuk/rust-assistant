"""
Exact deduplication for retrieval chunks.

This module implements ingest stage 1.7.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from rust_assistant.ingest.entities import Chunk

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

    def deduplicate(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Deduplicate chunks while preserving original order of kept entries.

        Args:
            chunks: Generated chunks before chunk-level deduplication.

        Returns:
            Deduplicated chunk list.
        """
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            key = (
                chunk.metadata.crate.value,
                self._normalized_text_hash(chunk.text),
            )
            groups[key].append(index)

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

        deduped = [chunk for index, chunk in enumerate(chunks) if index in keep_indexes]
        if duplicate_groups:
            logger.info(
                "Chunk dedup removed %s chunks across %s duplicate groups",
                duplicate_chunks,
                duplicate_groups,
            )
        return deduped


def deduplicate_chunks(
    chunks: list[Chunk],
    output_file: Optional[Union[Path, str]] = None,
) -> list[Chunk]:
    """
    Deduplicate chunks and optionally persist the result as JSONL.

    Args:
        chunks: Chunks produced by the chunking stage.
        output_file: Optional output path for deduplicated chunks.

    Returns:
        Deduplicated chunk list.
    """
    logger.info("Deduplicating %s chunks...", len(chunks))
    deduplicator = ChunkDeduplicator()
    deduped = deduplicator.deduplicate(chunks)
    logger.info("Chunk dedup complete: kept %s chunks", len(deduped))

    if output_file is not None:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for chunk in deduped:
                handle.write(chunk.model_dump_json() + "\n")
        logger.info("Saved deduplicated chunks to %s", out)

    return deduped
