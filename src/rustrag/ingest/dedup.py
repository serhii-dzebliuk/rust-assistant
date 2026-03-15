"""
Document-level deduplication.
Stage 1.5 of the ingest pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path

from ..models import Document

logger = logging.getLogger(__name__)


class DocumentDeduplicator:
    """Exact deduplication for parsed+cleaned documents."""

    def _normalized_text_hash(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _selection_key(self, doc: Document) -> tuple[int, int, int, str]:
        source = doc.source_path.replace("\\", "/")
        item_path = doc.metadata.item_path or ""
        return (source.count("/"), len(source), len(item_path), source)

    def deduplicate(self, docs: list[Document]) -> list[Document]:
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, doc in enumerate(docs):
            key = (doc.metadata.crate.value, self._normalized_text_hash(doc.text))
            groups[key].append(idx)

        keep_indexes: set[int] = set()
        duplicate_groups = 0
        duplicate_docs = 0
        for indexes in groups.values():
            if len(indexes) == 1:
                keep_indexes.add(indexes[0])
                continue

            duplicate_groups += 1
            duplicate_docs += len(indexes) - 1
            best_idx = min(indexes, key=lambda idx: self._selection_key(docs[idx]))
            keep_indexes.add(best_idx)

        deduped = [doc for idx, doc in enumerate(docs) if idx in keep_indexes]
        if duplicate_groups:
            logger.info(
                "Deduplicated documents: removed %s docs across %s duplicate groups",
                duplicate_docs,
                duplicate_groups,
            )
        return deduped


def deduplicate_documents(
    docs: list[Document],
    output_file: Path | str | None = None,
) -> list[Document]:
    logger.info("Deduplicating %s cleaned documents...", len(docs))
    deduplicator = DocumentDeduplicator()
    deduped = deduplicator.deduplicate(docs)
    logger.info("Dedup stage complete: kept %s docs", len(deduped))

    if output_file is not None:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for doc in deduped:
                f.write(doc.to_jsonl() + "\n")
        logger.info("Saved deduplicated documents to %s", out)

    return deduped
