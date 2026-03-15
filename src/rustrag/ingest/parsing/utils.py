from pathlib import Path

from rustrag.models import Crate, SourceType


def map_to_source_type(crate: Crate) -> SourceType:
    match crate:
        case Crate.BOOK: return SourceType.BOOK
        case Crate.CARGO: return SourceType.CARGO
        case Crate.REFERENCE: return SourceType.REFERENCE
        case Crate.STD | Crate.CORE | Crate.ALLOC | Crate.PROC_MACRO | Crate.TEST: return SourceType.RUSTDOC
        case _: return SourceType.RUSTDOC

def detect_crate_from_path(file_path: Path) -> Crate:
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
    file_abs = file_path.resolve()
    raw_abs = raw_data_dir.resolve()
    try:
        return str(file_abs.relative_to(raw_abs))
    except ValueError:
        return str(file_path)


def source_path_to_url(source_path: str, crate: Crate) -> str | None:
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
