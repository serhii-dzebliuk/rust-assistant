from pathlib import Path

import pytest

from rust_assistant.domain.enums import Crate
from rust_assistant.infrastructure.outbound.parsing.html.source_types import (
    ParserSourceType,
)
from rust_assistant.infrastructure.outbound.parsing.html.utils import (
    map_to_source_type,
    source_path_from_raw,
)

pytestmark = pytest.mark.unit


def test_source_path_from_raw_returns_posix_relative_path():
    raw_dir = Path("rust-docs")
    file_path = raw_dir / "std" / "primitive.unit.html"

    assert source_path_from_raw(raw_dir, file_path) == "std/primitive.unit.html"


def test_map_to_source_type_returns_parser_source_type():
    assert map_to_source_type(Crate.BOOK) == ParserSourceType.BOOK
    assert map_to_source_type(Crate.STD) == ParserSourceType.RUSTDOC
