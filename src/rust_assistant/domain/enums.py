"""Domain enum types."""

from __future__ import annotations

from enum import Enum


class ItemType(str, Enum):
    """Normalized Rust documentation item kinds."""

    FUNCTION = "fn"
    STRUCT = "struct"
    TRAIT = "trait"
    METHOD = "method"
    IMPL = "impl"
    MODULE = "module"
    MACRO = "macro"
    ENUM = "enum"
    CONSTANT = "constant"
    TYPE_ALIAS = "type"
    KEYWORD = "keyword"
    PRIMITIVE = "primitive"
    PAGE = "page"
    UNKNOWN = "unknown"


class Crate(str, Enum):
    """Supported documentation source identifiers."""

    STD = "std"
    CORE = "core"
    ALLOC = "alloc"
    PROC_MACRO = "proc_macro"
    TEST = "test"
    BOOK = "book"
    REFERENCE = "reference"
    RUSTC = "rustc"
    CARGO = "cargo"
    UNKNOWN = "unknown"

