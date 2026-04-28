"""Factory for source-specific HTML layouts."""

from rust_assistant.infrastructure.adapters.parsing.html.source_types import ParserSourceType

from .base_layout import HtmlLayout
from .book_layout import BookLayout
from .cargo_layout import CargoLayout
from .reference_layout import ReferenceLayout
from .rustdoc_layout import RustdocLayout


def get_layout(source_type: ParserSourceType) -> HtmlLayout:
    """Create an HTML layout strategy for a documentation source type."""
    match source_type:
        case ParserSourceType.BOOK:
            return BookLayout()
        case ParserSourceType.REFERENCE:
            return ReferenceLayout()
        case ParserSourceType.CARGO:
            return CargoLayout()
        case _:
            return RustdocLayout()
