"""Centralized runtime configuration for the rust-assistant package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Mapping, Optional

from dotenv import load_dotenv


TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


@dataclass(slots=True, frozen=True)
class Settings:
    """Full application settings tree."""

    app: AppSettings
    dependencies: DependencyStatusSettings
    postgres: PostgresSettings
    qdrant: QdrantSettings
    llm: LLMSettings
    logging: LoggingSettings
    proxy: ProxySettings


@dataclass(slots=True, frozen=True)
class AppSettings:
    """Application runtime settings."""

    host: str
    port: int
    reload: bool
    api_mode: str


@dataclass(slots=True, frozen=True)
class DependencyStatusSettings:
    """Dependency status values exposed by readiness endpoints."""

    postgres: str
    qdrant: str


@dataclass(slots=True, frozen=True)
class PostgresSettings:
    """PostgreSQL connection settings."""

    database: Optional[str]
    user: Optional[str]
    password: Optional[str]
    url: Optional[str]
    echo: bool
    pool_size: int
    max_overflow: int


@dataclass(slots=True, frozen=True)
class QdrantSettings:
    """Qdrant connection settings."""

    url: Optional[str]


@dataclass(slots=True, frozen=True)
class LLMSettings:
    """LLM and embedding provider settings."""

    provider: Optional[str]
    model: Optional[str]
    embedding_provider: Optional[str]
    embedding_model: Optional[str]


@dataclass(slots=True, frozen=True)
class LoggingSettings:
    """Logging settings."""

    level: str
    format: str


@dataclass(slots=True, frozen=True)
class ProxySettings:
    """Public/proxy settings used by local tooling and deployment."""

    public_base_url: Optional[str]


def build_settings(env: Mapping[str, str]) -> Settings:
    """Build the application settings tree from a mapping of environment values."""
    app = AppSettings(
        host=_read_str(env, "HOST", default="0.0.0.0"),
        port=_read_int(env, "PORT", default=8000),
        reload=_read_bool(env, "RELOAD", default=False),
        api_mode=_read_str(env, "API_MODE", default="stub"),
    )
    dependencies = DependencyStatusSettings(
        postgres=_read_str(env, "POSTGRES_STATUS", default="not_configured"),
        qdrant=_read_str(env, "QDRANT_STATUS", default="not_configured"),
    )
    postgres = PostgresSettings(
        database=_read_optional_str(env, "POSTGRES_DB"),
        user=_read_optional_str(env, "POSTGRES_USER"),
        password=_read_optional_str(env, "POSTGRES_PASSWORD"),
        url=_read_optional_str(env, "DATABASE_URL"),
        echo=_read_bool(env, "POSTGRES_ECHO", default=False),
        pool_size=_read_int(env, "POSTGRES_POOL_SIZE", default=10, minimum=0),
        max_overflow=_read_int(env, "POSTGRES_MAX_OVERFLOW", default=10, minimum=0),
    )
    qdrant = QdrantSettings(url=_read_optional_str(env, "QDRANT_URL"))
    llm = LLMSettings(
        provider=_read_optional_str(env, "LLM_PROVIDER"),
        model=_read_optional_str(env, "LLM_MODEL"),
        embedding_provider=_read_optional_str(env, "EMBEDDING_PROVIDER"),
        embedding_model=_read_optional_str(env, "EMBEDDING_MODEL"),
    )
    logging = LoggingSettings(
        level=_read_str(env, "LOG_LEVEL", default="INFO"),
        format=_read_str(env, "LOG_FORMAT", default="text"),
    )
    proxy = ProxySettings(
        public_base_url=_read_optional_str(env, "PUBLIC_BASE_URL"),
    )
    return Settings(
        app=app,
        dependencies=dependencies,
        postgres=postgres,
        qdrant=qdrant,
        llm=llm,
        logging=logging,
        proxy=proxy,
    )


def _read_str(
    env: Mapping[str, str],
    name: str,
    *,
    default: Optional[str] = None,
) -> str:
    """Read a required-or-default string value from a mapping."""
    value = env.get(name)
    if value is None or not value.strip():
        if default is None:
            raise ValueError(f"Environment variable {name} is required")
        return default
    return value.strip()


def _read_optional_str(env: Mapping[str, str], name: str) -> Optional[str]:
    """Read an optional string value from a mapping."""
    value = env.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _read_int(
    env: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int = 1,
) -> int:
    """Read and validate a positive integer value from a mapping."""
    raw_value = env.get(name)
    if raw_value is None or not raw_value.strip():
        value = default
    else:
        try:
            value = int(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be an integer") from exc

    if value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}")
    return value


def _read_bool(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    """Read and validate a boolean value from a mapping."""
    raw_value = env.get(name)
    if raw_value is None or not raw_value.strip():
        return default

    normalized = raw_value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"Environment variable {name} must be one of: "
        f"{', '.join(sorted(TRUE_VALUES | FALSE_VALUES))}"
    )


def load_settings() -> Settings:
    """Load environment variables from `.env` and build application settings."""
    load_dotenv(override=False)
    return build_settings(os.environ)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings for the current process."""
    return load_settings()
