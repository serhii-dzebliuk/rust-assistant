"""
Clean and normalize parsed documents.

This module implements ingest stage 1.4.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..models import Crate, Document

logger = logging.getLogger(__name__)


class DocumentCleaner:
    """
    Apply deterministic post-parse normalization rules.

    Responsibilities:
    - normalize whitespace/newlines
    - remove known textual artifacts
    - apply source-specific cleanup rules
    - drop extremely short low-value documents
    """

    MIN_TEXT_LENGTH = 30

    def clean_text(self, text: str, crate: Crate) -> str:
        """
        Normalize a single parsed document text.

        Args:
            text: Parsed text produced by stage 1.3.
            crate: Crate used for source-specific cleanup rules.

        Returns:
            Normalized text string.

        Example:
            >>> cleaner = DocumentCleaner()
            >>> cleaner.clean_text("A\\r\\n\\r\\nB", Crate.BOOK)
            'A\\n\\nB'
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.replace("Â§ ", "")
        text = text.replace("§ ", "")

        if crate == Crate.REFERENCE:
            text = re.sub(r"(?m)^\[[a-z0-9_.-]+\]\s*$\n?", "", text)

        text = re.sub(r"\b([A-Za-z0-9]+)_\s+([A-Za-z0-9]+)\b", r"\1_\2", text)
        text = re.sub(r"`([^`\n]+)`\s+s\b", r"`\1`s", text)

        return text.strip()

    def clean_document(self, doc: Document) -> Document | None:
        """
        Clean a document and optionally filter out too-short content.

        Args:
            doc: Parsed document before clean stage.

        Returns:
            Cleaned `Document` or `None` if document is below length threshold.
        """
        cleaned_text = self.clean_text(doc.text, doc.metadata.crate)
        if len(cleaned_text) < self.MIN_TEXT_LENGTH:
            return None
        return doc.model_copy(update={"text": cleaned_text})


def clean_documents(
    docs: list[Document],
    output_file: Path | str | None = None,
) -> list[Document]:
    """
    Clean a list of parsed documents.

    Args:
        docs: Parsed documents from stage 1.3.
        output_file: Optional JSONL path for cleaned output.

    Returns:
        List of cleaned documents.

    Example:
        >>> cleaned = clean_documents(parsed_docs)
        >>> len(cleaned) <= len(parsed_docs)
        True
    """
    logger.info("Cleaning %s documents...", len(docs))
    cleaner = DocumentCleaner()
    cleaned: list[Document] = []
    dropped_short = 0
    for doc in docs:
        cleaned_doc = cleaner.clean_document(doc)
        if cleaned_doc is None:
            dropped_short += 1
            continue
        cleaned.append(cleaned_doc)

    logger.info(
        "Clean stage complete: kept %s docs, dropped %s short docs",
        len(cleaned),
        dropped_short,
    )

    if output_file is not None:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for doc in cleaned:
                handle.write(doc.model_dump_json() + "\n")
        logger.info("Saved cleaned documents to %s", out)

    return cleaned
