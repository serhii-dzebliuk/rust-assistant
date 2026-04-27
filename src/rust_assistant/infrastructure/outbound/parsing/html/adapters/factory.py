"""Factory for source-specific HTML adapters."""

from rust_assistant.infrastructure.outbound.parsing.html.source_types import ParserSourceType

from .base_adapter import HtmlAdapter
from .book_adapter import BookAdapter
from .cargo_adapter import CargoAdapter
from .reference_adapter import ReferenceAdapter
from .rustdoc_adapter import RustdocAdapter


def get_adapter(source_type: ParserSourceType) -> HtmlAdapter:
    """
    Create an HTML adapter for a documentation source type.

    Args:
        source_type: Normalized source type detected from crate name.

    Returns:
        Adapter instance implementing source-specific extraction logic.

    Example:
        >>> adapter = get_adapter(ParserSourceType.BOOK)
        >>> adapter.__class__.__name__
        'BookAdapter'
    """
    match source_type:
        case ParserSourceType.BOOK:
            return BookAdapter()
        case ParserSourceType.REFERENCE:
            return ReferenceAdapter()
        case ParserSourceType.CARGO:
            return CargoAdapter()
        case _:
            return RustdocAdapter()
