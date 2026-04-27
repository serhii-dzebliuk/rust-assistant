from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_new_package_does_not_import_legacy_ingest_modules():
    legacy_imports: list[str] = []
    for path in Path("src/rust_assistant").rglob("*.py"):
        if "rust_assistant.ingest" in path.read_text(encoding="utf-8"):
            legacy_imports.append(path.as_posix())

    assert legacy_imports == []
