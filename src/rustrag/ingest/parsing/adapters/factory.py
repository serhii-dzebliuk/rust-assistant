"""Factory for source-specific HTML adapters."""

from rustrag.core.models import SourceType

from .base_adapter import HtmlAdapter
from .book_adapter import BookAdapter
from .cargo_adapter import CargoAdapter
from .reference_adapter import ReferenceAdapter
from .rustdoc_adapter import RustdocAdapter


def get_adapter(source_type: SourceType) -> HtmlAdapter:
    """
    Create an HTML adapter for a documentation source type.

    Args:
        source_type: Normalized source type detected from crate name.

    Returns:
        Adapter instance implementing source-specific extraction logic.

    Example:
        >>> adapter = get_adapter(SourceType.BOOK)
        >>> adapter.__class__.__name__
        'BookAdapter'
    """
    match source_type:
        case SourceType.BOOK:
            return BookAdapter()
        case SourceType.REFERENCE:
            return ReferenceAdapter()
        case SourceType.CARGO:
            return CargoAdapter()
        case _:
            return RustdocAdapter()
