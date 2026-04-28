import pytest

from rust_assistant.domain.enums import Crate
from rust_assistant.infrastructure.adapters.parsing.html.source_types import (
    ParserSourceType,
)
from rust_assistant.infrastructure.adapters.parsing.html.utils import map_to_source_type

pytestmark = pytest.mark.unit


def test_map_to_source_type_returns_parser_source_type():
    assert map_to_source_type(Crate.BOOK) == ParserSourceType.BOOK
    assert map_to_source_type(Crate.STD) == ParserSourceType.RUSTDOC
