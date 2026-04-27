"""Structured-text rendering helpers used by domain policies."""

from __future__ import annotations

from collections.abc import Sequence
import re

from rust_assistant.domain.value_objects.structured_blocks import (
    BlockType,
    StructuredBlock,
)


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and punctuation spacing in rendered text.

    Args:
        text: Raw text blocks joined from structured content.

    Returns:
        Cleaned text with normalized newlines and spacing.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text.strip()


def blocks_to_text(blocks: Sequence[StructuredBlock]) -> str:
    """
    Render structured blocks into normalized markdown-like text.

    Args:
        blocks: Structured blocks extracted from the source document.

    Returns:
        Plain text representation used by downstream domain policies.
    """

    rendered_blocks: list[str] = []
    for block in blocks:
        if block.block_type == BlockType.HEADING:
            rendered_blocks.append(block.text)
            continue
        if block.block_type == BlockType.PARAGRAPH:
            rendered_blocks.append(block.text)
            continue
        if block.block_type == BlockType.LIST_ITEM:
            prefix = "  " * (block.list_depth or 0) + "- "
            rendered_blocks.append(f"{prefix}{block.text}")
            continue
        if block.block_type == BlockType.CODE_BLOCK:
            language = block.code_language or "text"
            rendered_blocks.append(f"```{language}\n{block.text}\n```")
            continue
        if block.block_type == BlockType.DEFINITION_TERM:
            rendered_blocks.append(f"- {block.text}")
            continue
        if block.block_type == BlockType.DEFINITION_DESC:
            rendered_blocks.append(f"  - {block.text}")

    return normalize_text("\n\n".join(rendered_blocks))
