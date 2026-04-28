import shutil
import uuid
from collections.abc import Generator
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
    from rust_assistant.infrastructure.adapters.parsing.html.document_parser import (
        HtmlDocumentParser,
    )

    return HtmlDocumentParser(raw_data_dir=raw_data_dir)


@pytest.fixture
def workspace_tmp_path() -> Generator[Path, None, None]:
    """Return a temporary directory rooted inside the workspace."""
    base_dir = Path("test_tmp").resolve()
    base_dir.mkdir(exist_ok=True)
    temp_dir = base_dir / uuid.uuid4().hex
    temp_dir.mkdir()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
