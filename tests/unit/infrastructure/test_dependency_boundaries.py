from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_infrastructure_modules_do_not_import_bootstrap_settings_directly():
    repo_root = Path(__file__).resolve().parents[3]
    infrastructure_roots = [
        repo_root / "src" / "rust_assistant" / "infrastructure",
    ]

    for infrastructure_root in infrastructure_roots:
        if not infrastructure_root.exists():
            continue
        for path in infrastructure_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "rust_assistant.bootstrap" not in source
