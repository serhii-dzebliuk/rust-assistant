"""Document cleaning policy functions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import re
from typing import Optional

from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate
from rust_assistant.domain.policies.text_rendering import blocks_to_text
from rust_assistant.domain.value_objects.structured_blocks import (
    BlockType,
    StructuredBlock,
)


_MIN_DOCUMENT_TEXT_LENGTH = 30
_SECTION_ARTIFACT_MARKERS = ("§ ",)
_SECTION_ARTIFACT_PATTERN = re.compile(r"^[^\s]{0,8}§\s+")


def clean_text(text: str, crate: Crate) -> str:
    """
    Normalize one parsed document text.

    Args:
        text: Parsed text produced by the parsing stage.
        crate: Crate used for source-specific cleanup rules.

    Returns:
        Normalized text string.
    """

    cleaned_text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_text = re.sub(r"[ \t]+\n", "\n", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = _strip_section_artifacts(cleaned_text)

    if crate == Crate.REFERENCE:
        cleaned_text = re.sub(r"(?m)^\[[a-z0-9_.-]+\]\s*$\n?", "", cleaned_text)

    cleaned_text = re.sub(r"\b([A-Za-z0-9]+)_\s+([A-Za-z0-9]+)\b", r"\1_\2", cleaned_text)
    cleaned_text = re.sub(r"`([^`\n]+)`\s+s\b", r"`\1`s", cleaned_text)
    cleaned_text = re.sub(r" +([,.;:!?])", r"\1", cleaned_text)

    return cleaned_text.strip()


def clean_code_text(text: str) -> str:
    """
    Normalize code block text without collapsing code formatting.

    Args:
        text: Raw code block text.

    Returns:
        Cleaned code text preserving line structure.
    """

    cleaned_text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_text = _strip_section_artifacts(cleaned_text)
    return cleaned_text.strip("\n")


def clean_blocks(
    blocks: Sequence[StructuredBlock],
    crate: Crate,
) -> list[StructuredBlock]:
    """
    Normalize structured block text while preserving block boundaries.

    Args:
        blocks: Structured blocks extracted during parsing.
        crate: Crate used for source-specific cleanup rules.

    Returns:
        Cleaned structured blocks with empty blocks removed.
    """

    cleaned_blocks: list[StructuredBlock] = []
    for block in blocks:
        if block.block_type == BlockType.CODE_BLOCK:
            cleaned_text = clean_code_text(block.text)
        else:
            cleaned_text = clean_text(block.text, crate)
        if not cleaned_text:
            continue
        cleaned_blocks.append(replace(block, text=cleaned_text))
    return cleaned_blocks


def clean_document(
    document: Document,
    min_text_length: int = _MIN_DOCUMENT_TEXT_LENGTH,
) -> Optional[Document]:
    """
    Clean a document and optionally filter out too-short content.

    Args:
        document: Parsed document before the clean stage.
        min_text_length: Minimum acceptable document length after normalization.

    Returns:
        Cleaned document or `None` if the document is below the length threshold.
    """

    cleaned_blocks = clean_blocks(document.structured_blocks, document.crate)
    if cleaned_blocks:
        cleaned_text = blocks_to_text(cleaned_blocks)
    else:
        cleaned_text = clean_text(document.text, document.crate)

    if len(cleaned_text) < min_text_length:
        return None

    return replace(
        document,
        text=cleaned_text,
        structured_blocks=cleaned_blocks,
    )


def clean_documents(
    documents: list[Document],
    min_text_length: int = _MIN_DOCUMENT_TEXT_LENGTH,
) -> list[Document]:
    """
    Clean a list of parsed documents.

    Args:
        documents: Parsed documents from the parsing stage.
        min_text_length: Minimum acceptable document length after normalization.

    Returns:
        List of cleaned documents.
    """

    cleaned_documents: list[Document] = []
    for document in documents:
        cleaned_document = clean_document(document, min_text_length=min_text_length)
        if cleaned_document is None:
            continue
        cleaned_documents.append(cleaned_document)
    return cleaned_documents


def _strip_section_artifacts(text: str) -> str:
    cleaned_lines = []
    for line in text.split("\n"):
        cleaned_line = _SECTION_ARTIFACT_PATTERN.sub("", line)
        for marker in _SECTION_ARTIFACT_MARKERS:
            cleaned_line = cleaned_line.replace(marker, "")
        cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)
