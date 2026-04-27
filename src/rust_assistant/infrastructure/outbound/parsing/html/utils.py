"""Utility helpers for parsing stage path and source mapping."""

from pathlib import Path
from typing import Optional

from rust_assistant.domain.enums import Crate
from rust_assistant.infrastructure.outbound.parsing.html.source_types import ParserSourceType


def map_to_source_type(crate: Crate) -> ParserSourceType:
    """
    Map a crate identifier to a parser source type.

    Args:
        crate: Detected crate for a file path.

    Returns:
        Source type used by the adapter factory.

    Example:
        >>> map_to_source_type(Crate.BOOK)
        <ParserSourceType.BOOK: 'book'>
    """
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
    """
    Detect crate name from a file path under the configured raw docs root.

    Args:
        file_path: Absolute or relative path to an HTML file.

    Returns:
        Detected crate enum, or `Crate.UNKNOWN` if detection fails.

    Example:
        >>> detect_crate_from_path(Path("rust-docs/std/alloc/index.html"))
        <Crate.STD: 'std'>
    """
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


def source_path_from_raw(raw_data_dir: Path, file_path: Path) -> str:
    """
    Build a path relative to the raw data root.

    Args:
        raw_data_dir: Root directory for raw HTML files.
        file_path: Absolute or relative source file path.

    Returns:
        Relative source path string used in document metadata.

    Example:
        >>> source_path_from_raw(Path("rust-docs"), Path("rust-docs/book/index.html"))
        'book/index.html'
    """
    file_abs = file_path.resolve()
    raw_abs = raw_data_dir.resolve()
    try:
        return file_abs.relative_to(raw_abs).as_posix()
    except ValueError:
        return file_path.as_posix()


def source_path_to_url(source_path: str, crate: Crate) -> Optional[str]:
    """
    Build canonical online documentation URL for a parsed source path.

    Args:
        source_path: Relative source path like `std/alloc/index.html`.
        crate: Crate used to select base URL.

    Returns:
        Absolute documentation URL or `None` for unsupported crates.

    Example:
        >>> source_path_to_url("std/alloc/index.html", Crate.STD)
        'https://doc.rust-lang.org/std/alloc/index.html'
    """
    normalized = source_path.replace("\\", "/")
    if "/" not in normalized:
        return None

    _, relative_path = normalized.split("/", 1)
    base_urls = {
        Crate.BOOK: "https://doc.rust-lang.org/book/",
        Crate.REFERENCE: "https://doc.rust-lang.org/reference/",
        Crate.CARGO: "https://doc.rust-lang.org/cargo/",
        Crate.STD: "https://doc.rust-lang.org/std/",
        Crate.CORE: "https://doc.rust-lang.org/core/",
        Crate.ALLOC: "https://doc.rust-lang.org/alloc/",
        Crate.PROC_MACRO: "https://doc.rust-lang.org/proc_macro/",
        Crate.TEST: "https://doc.rust-lang.org/test/",
    }
    base_url = base_urls.get(crate)
    if base_url is None:
        return None
    return base_url + relative_path
