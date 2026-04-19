from pathlib import Path

import pytest

from rust_assistant.ingest.parsing.utils import source_path_from_raw

pytestmark = pytest.mark.unit


def test_source_path_from_raw_returns_posix_relative_path():
    raw_dir = Path("rust-docs")
    file_path = raw_dir / "std" / "primitive.unit.html"

    assert source_path_from_raw(raw_dir, file_path) == "std/primitive.unit.html"
