import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[3] / "alembic" / "versions" / "0002_uuid_business_ids.py"
)
_SPEC = importlib.util.spec_from_file_location("uuid_business_ids_migration", _MIGRATION_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


def test_build_document_uuid_is_stable():
    first = migration.build_document_uuid("std/keyword.async.html")
    second = migration.build_document_uuid("std/keyword.async.html")

    assert first == second


def test_build_chunk_uuid_is_stable():
    first = migration.build_chunk_uuid("std/keyword.async.html", 0)
    second = migration.build_chunk_uuid("std/keyword.async.html", 0)

    assert first == second


def test_build_chunk_uuid_changes_with_chunk_index():
    first = migration.build_chunk_uuid("std/keyword.async.html", 0)
    second = migration.build_chunk_uuid("std/keyword.async.html", 1)

    assert first != second
