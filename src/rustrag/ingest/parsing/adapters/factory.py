from rustrag.ingest.parsing.adapters.base_adapter import HtmlAdapter
from rustrag.ingest.parsing.adapters.book_adapter import BookAdapter
from rustrag.ingest.parsing.adapters.cargo_adapter import CargoAdapter
from rustrag.ingest.parsing.adapters.reference_adapter import ReferenceAdapter
from rustrag.ingest.parsing.adapters.rustdoc_adapter import RustdocAdapter
from rustrag.models import SourceType


def get_adapter(source_type: SourceType) -> HtmlAdapter:
    match source_type:
        case SourceType.BOOK: return BookAdapter()
        case SourceType.REFERENCE: return ReferenceAdapter()
        case SourceType.CARGO: return CargoAdapter()
        case _: return RustdocAdapter()
