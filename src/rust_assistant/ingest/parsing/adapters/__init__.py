"""Source-specific parsing adapters."""

from .base_adapter import HtmlAdapter
from .book_adapter import BookAdapter
from .cargo_adapter import CargoAdapter
from .reference_adapter import ReferenceAdapter
from .rustdoc_adapter import RustdocAdapter

__all__ = [
    "HtmlAdapter",
    "BookAdapter",
    "CargoAdapter",
    "ReferenceAdapter",
    "RustdocAdapter",
]
