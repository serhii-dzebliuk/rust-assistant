from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def raw_data_dir() -> Path:
    """Return the repository fixture directory with raw Rust docs."""
    return Path("data/raw")


@pytest.fixture
def page_parser(raw_data_dir: Path) -> Any:
    """Build a parser for integration tests that exercise real sample pages."""
    from rust_assistant.ingest.parsing.page_parser import PageParser

    return PageParser(raw_data_dir=raw_data_dir)
