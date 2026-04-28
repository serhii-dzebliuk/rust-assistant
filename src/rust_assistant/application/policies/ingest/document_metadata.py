"""Document metadata normalization policy for ingest parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rust_assistant.domain.enums import Crate, ItemType


RUSTDOC_CRATES = {
    Crate.STD,
    Crate.CORE,
    Crate.ALLOC,
    Crate.PROC_MACRO,
    Crate.TEST,
}

ITEM_TYPE_PATTERNS = {
    ItemType.FUNCTION: (r"\bfn\b", r"\bfunction\b"),
    ItemType.STRUCT: (r"\bstruct\b",),
    ItemType.TRAIT: (r"\btrait\b",),
    ItemType.METHOD: (r"\bmethod\b",),
    ItemType.IMPL: (r"\bimpl\b", r"\bimplementation\b"),
    ItemType.MODULE: (r"\bmod\b", r"\bmodule\b"),
    ItemType.MACRO: (r"\bmacro\b",),
    ItemType.ENUM: (r"\benum\b",),
    ItemType.CONSTANT: (r"\bconst\b", r"\bconstant\b"),
    ItemType.TYPE_ALIAS: (r"\btype\b",),
}

RUSTDOC_FILE_PREFIX_TO_ITEM_TYPE = {
    "fn.": ItemType.FUNCTION,
    "struct.": ItemType.STRUCT,
    "trait.": ItemType.TRAIT,
    "macro.": ItemType.MACRO,
    "enum.": ItemType.ENUM,
    "constant.": ItemType.CONSTANT,
    "type.": ItemType.TYPE_ALIAS,
    "mod.": ItemType.MODULE,
}

RUSTDOC_BODY_CLASS_TO_ITEM_TYPE = {
    "mod": ItemType.MODULE,
    "struct": ItemType.STRUCT,
    "trait": ItemType.TRAIT,
    "macro": ItemType.MACRO,
    "enum": ItemType.ENUM,
    "constant": ItemType.CONSTANT,
    "type": ItemType.TYPE_ALIAS,
    "function": ItemType.FUNCTION,
    "fn": ItemType.FUNCTION,
    "method": ItemType.METHOD,
    "impl": ItemType.IMPL,
}


@dataclass(slots=True, frozen=True)
class ParsedDocumentFacts:
    """Plain facts used to derive canonical parsed document metadata."""

    raw_data_dir: Path
    file_path: Path
    crate: Crate
    title: str
    text: str
    breadcrumbs: tuple[str, ...] = ()
    rustdoc_body_classes: tuple[str, ...] = ()


def source_path_from_raw(raw_data_dir: Path, file_path: Path) -> str:
    """Build a normalized source path relative to the raw docs root."""
    file_abs = file_path.resolve()
    raw_abs = raw_data_dir.resolve()
    try:
        return file_abs.relative_to(raw_abs).as_posix()
    except ValueError:
        return file_path.as_posix()


def source_path_to_url(source_path: str, crate: Crate) -> Optional[str]:
    """Build the canonical online documentation URL for a parsed source path."""
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


def build_item_path(facts: ParsedDocumentFacts) -> Optional[str]:
    """Build stable item path metadata for a parsed document."""
    if facts.crate in RUSTDOC_CRATES:
        return facts.title or None

    if facts.breadcrumbs:
        return "::".join(facts.breadcrumbs)

    try:
        rel = facts.file_path.resolve().relative_to(
            (facts.raw_data_dir / facts.crate.value).resolve()
        )
    except ValueError:
        return None

    parts = list(rel.parts)
    if not parts:
        return None
    parts[-1] = rel.stem
    parts = [part for part in parts if part not in {"index", "."}]
    if not parts:
        return facts.crate.value
    return f"{facts.crate.value}::" + "::".join(parts)


def detect_item_type(facts: ParsedDocumentFacts) -> Optional[ItemType]:
    """Detect normalized document item type metadata."""
    if facts.crate in {Crate.BOOK, Crate.CARGO, Crate.REFERENCE}:
        return ItemType.PAGE
    if facts.crate != Crate.STD:
        return None

    rustdoc_item_type = _detect_rustdoc_item_type(facts)
    if rustdoc_item_type is not None:
        return rustdoc_item_type

    title_l = facts.title.lower()
    for item_type, patterns in ITEM_TYPE_PATTERNS.items():
        if any(re.search(pattern, title_l, re.IGNORECASE) for pattern in patterns):
            return item_type

    sample = facts.text[:700].lower()
    for item_type, patterns in ITEM_TYPE_PATTERNS.items():
        if any(re.search(pattern, sample, re.IGNORECASE) for pattern in patterns):
            return item_type
    return ItemType.UNKNOWN


def _detect_rustdoc_item_type(facts: ParsedDocumentFacts) -> Optional[ItemType]:
    """Detect rustdoc item type using file naming and extracted HTML facts."""
    stem = facts.file_path.stem
    for prefix, item_type in RUSTDOC_FILE_PREFIX_TO_ITEM_TYPE.items():
        if stem.startswith(prefix):
            return item_type

    if facts.file_path.name == "index.html":
        return ItemType.MODULE

    for css_class in facts.rustdoc_body_classes:
        item_type = RUSTDOC_BODY_CLASS_TO_ITEM_TYPE.get(css_class)
        if item_type is not None:
            return item_type

    title_l = facts.title.lower()
    if title_l == "std":
        return ItemType.MODULE

    item_name = facts.title.rsplit("::", 1)[-1]
    if item_name.endswith("!"):
        return ItemType.MACRO
    if "::keyword::" in title_l:
        return ItemType.KEYWORD
    if "::primitive::" in title_l:
        return ItemType.PRIMITIVE
    return None
