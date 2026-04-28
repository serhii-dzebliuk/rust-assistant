import pytest

from rust_assistant.infrastructure.adapters.sqlalchemy.config import SqlAlchemyConfig
from rust_assistant.infrastructure.adapters.sqlalchemy.session import (
    build_async_engine,
    build_session_factory,
)

pytestmark = pytest.mark.unit


def test_build_async_engine_returns_none_without_database_url():
    config = SqlAlchemyConfig(url=None, echo=False, pool_size=10, max_overflow=10)

    engine = build_async_engine(config)
    session_factory = build_session_factory(engine)

    assert engine is None
    assert session_factory is None
