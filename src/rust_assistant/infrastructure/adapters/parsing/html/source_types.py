"""Parser adapter source-type enum."""

from __future__ import annotations

from enum import Enum


class ParserSourceType(str, Enum):
    """High-level source layout used by the parsing adapter factory."""

    BOOK = "book"
    REFERENCE = "reference"
    CARGO = "cargo"
    RUSTDOC = "rustdoc"
