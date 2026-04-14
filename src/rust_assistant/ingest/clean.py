"""
Clean and normalize parsed documents.

This module implements ingest stage 1.4.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union

from rust_assistant.ingest.entities import BlockType, Document, StructuredBlock
from rust_assistant.schemas.enums import Crate

from .parsing.core import blocks_to_text

logger = logging.getLogger(__name__)


class DocumentCleaner:
    """
    Apply deterministic post-parse normalization rules.

    Responsibilities:
    - normalize whitespace/newlines
    - remove known textual artifacts
    - apply source-specific cleanup rules
    - drop extremely short low-value documents
    - preserve structured blocks for downstream chunking
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
        for marker in ("Ã‚Â§ ", "Â§ ", "§ ", "ｧ "):
            text = text.replace(marker, "")

        if crate == Crate.REFERENCE:
            text = re.sub(r"(?m)^\[[a-z0-9_.-]+\]\s*$\n?", "", text)

        text = re.sub(r"\b([A-Za-z0-9]+)_\s+([A-Za-z0-9]+)\b", r"\1_\2", text)
        text = re.sub(r"`([^`\n]+)`\s+s\b", r"`\1`s", text)
        text = re.sub(r" +([,.;:!?])", r"\1", text)

        return text.strip()

    def clean_code_text(self, text: str) -> str:
        """
        Normalize code block text without collapsing code formatting.

        Args:
            text: Raw code block text.

        Returns:
            Cleaned code text preserving line structure.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for marker in ("Ã‚Â§ ", "Â§ ", "§ ", "ｧ "):
            text = text.replace(marker, "")
        return text.strip("\n")

    def clean_blocks(
        self,
        blocks: list[StructuredBlock],
        crate: Crate,
    ) -> list[StructuredBlock]:
        """
        Normalize structured block text while preserving block boundaries.

        Args:
            blocks: Structured blocks extracted during parse stage.
            crate: Crate used for source-specific cleanup rules.

        Returns:
            Cleaned structured blocks with empty blocks removed.
        """
        cleaned_blocks: list[StructuredBlock] = []
        for block in blocks:
            if block.block_type == BlockType.CODE_BLOCK:
                cleaned_text = self.clean_code_text(block.text)
            else:
                cleaned_text = self.clean_text(block.text, crate)
            if not cleaned_text:
                continue
            cleaned_blocks.append(block.model_copy(update={"text": cleaned_text}))
        return cleaned_blocks

    def clean_document(self, doc: Document) -> Optional[Document]:
        """
        Clean a document and optionally filter out too-short content.

        Args:
            doc: Parsed document before clean stage.

        Returns:
            Cleaned `Document` or `None` if document is below length threshold.
        """
        cleaned_blocks = self.clean_blocks(doc.structured_blocks, doc.metadata.crate)
        if cleaned_blocks:
            cleaned_text = blocks_to_text(cleaned_blocks)
        else:
            cleaned_text = self.clean_text(doc.text, doc.metadata.crate)
        if len(cleaned_text) < self.MIN_TEXT_LENGTH:
            return None
        return doc.model_copy(
            update={
                "text": cleaned_text,
                "structured_blocks": cleaned_blocks,
            }
        )


def clean_documents(
    docs: list[Document],
    output_file: Optional[Union[Path, str]] = None,
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
