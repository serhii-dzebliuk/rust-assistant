import pytest

from rust_assistant.core.config import PostgresSettings
from rust_assistant.core.db import build_async_engine, build_session_factory

pytestmark = pytest.mark.unit


def test_build_async_engine_returns_none_without_database_url():
    settings = PostgresSettings(
        database=None,
        user=None,
        password=None,
        url=None,
        echo=False,
        pool_size=10,
        max_overflow=10,
    )

    engine = build_async_engine(settings)
    session_factory = build_session_factory(engine)

    assert engine is None
    assert session_factory is None
