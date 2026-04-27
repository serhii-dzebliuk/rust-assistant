"""Composition-root helpers that turn global settings into concrete runtime wiring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from rust_assistant.bootstrap.logging import configure_logging
from rust_assistant.bootstrap.settings import LoggingSettings, Settings, get_settings
from rust_assistant.infrastructure.outbound.sqlalchemy.config import SqlAlchemyConfig


@dataclass(slots=True, frozen=True)
class RuntimeContainer:
    """Resolved runtime dependencies for current entrypoints."""

    settings: Settings
    sqlalchemy: SqlAlchemyConfig


def _build_sqlalchemy_config(settings: Settings) -> SqlAlchemyConfig:
    """Map global app settings into the SQLAlchemy adapter configuration."""
    postgres = settings.postgres
    return SqlAlchemyConfig(
        url=postgres.url,
        echo=postgres.echo,
        pool_size=postgres.pool_size,
        max_overflow=postgres.max_overflow,
    )


def build_container(
    *,
    settings: Optional[Settings] = None,
    logging_settings: Optional[LoggingSettings] = None,
) -> RuntimeContainer:
    """Load runtime settings, configure logging, and wire adapter configuration."""
    runtime_settings = settings or get_settings()
    configure_logging(logging_settings=logging_settings or runtime_settings.logging)
    return RuntimeContainer(
        settings=runtime_settings,
        sqlalchemy=_build_sqlalchemy_config(runtime_settings),
    )


def build_container_with_log_level(
    *,
    settings: Optional[Settings] = None,
    log_level: Optional[str] = None,
) -> RuntimeContainer:
    """Build the runtime container with an optional temporary log-level override."""
    runtime_settings = settings or get_settings()
    logging_settings = runtime_settings.logging
    if log_level is not None:
        logging_settings = replace(logging_settings, level=log_level)
    return build_container(settings=runtime_settings, logging_settings=logging_settings)
