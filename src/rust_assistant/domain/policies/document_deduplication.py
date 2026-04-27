"""Document deduplication policy functions."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from rust_assistant.domain.entities.documents import Document


def _normalized_text_hash(text: str) -> str:
    """
    Create a stable hash of normalized text.

    Args:
        text: Input document text.

    Returns:
        SHA-256 hexadecimal digest of normalized text.
    """

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _document_selection_key(document: Document) -> tuple[int, int, int, str]:
    """
    Build the sort key used to pick the canonical document from duplicates.

    Args:
        document: Candidate document in a duplicate group.

    Returns:
        Tuple used for deterministic canonical selection.
    """

    source_path = document.source_path.replace("\\", "/")
    item_path = document.item_path or ""
    return (source_path.count("/"), len(source_path), len(item_path), source_path)


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """
    Deduplicate documents and keep one canonical entry per exact group.

    Args:
        documents: Cleaned documents.

    Returns:
        Deduplicated document list preserving original order of kept entries.
    """

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        key = (document.crate.value, _normalized_text_hash(document.text))
        groups[key].append(index)

    keep_indexes: set[int] = set()
    for indexes in groups.values():
        if len(indexes) == 1:
            keep_indexes.add(indexes[0])
            continue

        best_index = min(indexes, key=lambda current: _document_selection_key(documents[current]))
        keep_indexes.add(best_index)

    return [document for index, document in enumerate(documents) if index in keep_indexes]
