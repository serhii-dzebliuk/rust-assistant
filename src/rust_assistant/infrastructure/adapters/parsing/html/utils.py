"""Utility helpers for parsing source layout mapping."""

from pathlib import Path

from rust_assistant.domain.enums import Crate
from rust_assistant.infrastructure.adapters.parsing.html.source_types import ParserSourceType


def map_to_source_type(crate: Crate) -> ParserSourceType:
    """Map a crate identifier to a parser source layout type."""
    match crate:
        case Crate.BOOK:
            return ParserSourceType.BOOK
        case Crate.CARGO:
            return ParserSourceType.CARGO
        case Crate.REFERENCE:
            return ParserSourceType.REFERENCE
        case Crate.STD | Crate.CORE | Crate.ALLOC | Crate.PROC_MACRO | Crate.TEST:
            return ParserSourceType.RUSTDOC
        case _:
            return ParserSourceType.RUSTDOC


def detect_crate_from_path(file_path: Path) -> Crate:
    """Detect crate name from a file path under the configured raw docs root."""
    try:
        parts = file_path.parts
    except AttributeError:
        return Crate.UNKNOWN

    if not parts:
        return Crate.UNKNOWN

    try:
        raw_index = parts.index("raw")
        prefix = parts[raw_index + 1].lower()
    except (ValueError, IndexError):
        prefix = parts[0].lower()

    if prefix in {"book", "cargo", "reference", "std"}:
        return Crate(prefix)
    return Crate.UNKNOWN
