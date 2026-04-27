"""Structured block value objects used by parsed documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BlockType(str, Enum):
    """Normalized structured content blocks extracted from HTML."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    DEFINITION_TERM = "definition_term"
    DEFINITION_DESC = "definition_desc"


@dataclass(slots=True, frozen=True)
class StructuredBlock:
    """Structured representation of one parsed content block."""

    block_type: BlockType
    text: str
    html_tag: str
    heading_level: Optional[int] = None
    list_depth: Optional[int] = None
    code_language: Optional[str] = None
    anchor: Optional[str] = None
    section_path: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize section path into an immutable tuple."""

        object.__setattr__(self, "section_path", tuple(self.section_path))
