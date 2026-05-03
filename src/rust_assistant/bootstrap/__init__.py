"""Composition-root helpers for the migrated clean architecture."""

from rust_assistant.bootstrap.api import create_app
from rust_assistant.bootstrap.container import (
    RuntimeContainer,
    build_container,
    build_container_with_log_level,
)
from rust_assistant.bootstrap.logging import JsonFormatter, configure_logging
from rust_assistant.bootstrap.settings import (
    AppSettings,
    ChatSettings,
    EmbeddingSettings,
    IngestSettings,
    LoggingSettings,
    OpenAISettings,
    PostgresSettings,
    ProxySettings,
    QdrantSettings,
    Settings,
    build_settings,
    get_settings,
    load_settings,
)

__all__ = [
    "AppSettings",
    "ChatSettings",
    "EmbeddingSettings",
    "IngestSettings",
    "JsonFormatter",
    "LoggingSettings",
    "OpenAISettings",
    "PostgresSettings",
    "ProxySettings",
    "QdrantSettings",
    "RuntimeContainer",
    "Settings",
    "build_container",
    "build_container_with_log_level",
    "build_settings",
    "configure_logging",
    "create_app",
    "get_settings",
    "load_settings",
]
