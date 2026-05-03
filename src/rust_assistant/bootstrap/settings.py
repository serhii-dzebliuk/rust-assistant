"""Centralized runtime settings for the migrated composition root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final, Mapping, Optional

from dotenv import load_dotenv

TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


@dataclass(slots=True, frozen=True)
class Settings:
    """Full application settings tree."""

    app: AppSettings
    postgres: PostgresSettings
    qdrant: QdrantSettings
    openai: OpenAISettings
    chat: ChatSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    ingest: IngestSettings
    logging: LoggingSettings
    proxy: ProxySettings


@dataclass(slots=True, frozen=True)
class AppSettings:
    """Application runtime settings."""

    host: str
    port: int


@dataclass(slots=True, frozen=True)
class OpenAISettings:
    """OpenAI provider settings."""

    model: Optional[str]
    api_key: Optional[str] = field(repr=False)
    max_output_tokens: int
    temperature: float
    request_timeout_seconds: float


@dataclass(slots=True, frozen=True)
class PostgresSettings:
    """PostgreSQL connection settings loaded from environment variables."""

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
    collection_name: str
    vector_size: Optional[int]
    distance: str
    upsert_batch_size: int


@dataclass(slots=True, frozen=True)
class ChatSettings:
    """Chat and RAG behavior settings."""

    retrieval_limit: int
    reranking_limit: int
    use_reranking: bool
    max_query_tokens: int
    max_context_tokens: int


@dataclass(slots=True, frozen=True)
class EmbeddingSettings:
    """Embedding model and serving settings."""

    model: Optional[str]
    base_url: Optional[str]
    normalize: bool
    pooling: str
    max_batch_tokens: int
    max_batch_items: int
    max_concurrent_requests: int
    request_timeout_seconds: float


@dataclass(slots=True, frozen=True)
class RerankerSettings:
    """Reranker model and serving settings."""

    model: Optional[str]
    base_url: Optional[str]
    max_batch_items: int


@dataclass(slots=True, frozen=True)
class IngestSettings:
    """Ingest pipeline settings."""

    raw_docs_dir: Optional[Path]
    max_chunk_chars: int
    min_chunk_chars: int


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
    qdrant = QdrantSettings(
        url=_read_optional_str(env, "QDRANT_URL"),
        collection_name=_read_str(env, "QDRANT_COLLECTION_NAME", default="rust-docs"),
        vector_size=_read_optional_int(env, "QDRANT_VECTOR_SIZE", minimum=1),
        distance=_read_str(env, "QDRANT_DISTANCE", default="cosine"),
        upsert_batch_size=_read_int(env, "QDRANT_UPSERT_BATCH_SIZE", default=256),
    )
    openai = OpenAISettings(
        model=_read_optional_str(env, "OPENAI_MODEL"),
        api_key=_read_optional_str(env, "OPENAI_API_KEY"),
        max_output_tokens=_read_int(env, "OPENAI_MAX_OUTPUT_TOKENS", default=500),
        temperature=_read_non_negative_float(env, "OPENAI_TEMPERATURE", default=0.2),
        request_timeout_seconds=_read_float(
            env,
            "OPENAI_REQUEST_TIMEOUT_SECONDS",
            default=60.0,
        ),
    )
    chat = ChatSettings(
        retrieval_limit=_read_int(env, "CHAT_RETRIEVAL_LIMIT", default=20),
        reranking_limit=_read_int(env, "CHAT_RERANKING_LIMIT", default=5),
        use_reranking=_read_bool(env, "CHAT_USE_RERANKING", default=True),
        max_query_tokens=_read_int(env, "CHAT_MAX_QUERY_TOKENS", default=1000),
        max_context_tokens=_read_int(env, "CHAT_MAX_CONTEXT_TOKENS", default=2500),
    )
    if chat.reranking_limit > chat.retrieval_limit:
        raise ValueError("CHAT_RERANKING_LIMIT must be <= CHAT_RETRIEVAL_LIMIT")
    embedding = EmbeddingSettings(
        model=_read_optional_str(env, "EMBEDDING_MODEL"),
        base_url=_read_optional_str(env, "EMBEDDING_BASE_URL"),
        normalize=_read_bool(env, "EMBEDDING_NORMALIZE", default=True),
        pooling=_read_str(env, "EMBEDDING_POOLING", default="mean"),
        max_batch_tokens=_read_int(env, "EMBEDDING_MAX_BATCH_TOKENS", default=4096),
        max_batch_items=_read_int(env, "EMBEDDING_MAX_BATCH_ITEMS", default=64),
        max_concurrent_requests=_read_int(
            env,
            "EMBEDDING_MAX_CONCURRENT_REQUESTS",
            default=8,
        ),
        request_timeout_seconds=_read_float(
            env,
            "EMBEDDING_REQUEST_TIMEOUT_SECONDS",
            default=120.0,
        ),
    )
    reranker = RerankerSettings(
        model=_read_optional_str(env, "RERANKER_MODEL"),
        base_url=_read_optional_str(env, "RERANKER_BASE_URL"),
        max_batch_items=_read_int(env, "RERANKER_MAX_BATCH_ITEMS", default=32),
    )
    ingest = IngestSettings(
        raw_docs_dir=_read_optional_path(env, "RUST_DOCS_RAW_DIR"),
        max_chunk_chars=_read_int(env, "INGEST_MAX_CHUNK_CHARS", default=1400),
        min_chunk_chars=_read_int(env, "INGEST_MIN_CHUNK_CHARS", default=180),
    )
    if ingest.min_chunk_chars > ingest.max_chunk_chars:
        raise ValueError("INGEST_MIN_CHUNK_CHARS must be <= INGEST_MAX_CHUNK_CHARS")
    logging = LoggingSettings(
        level=_read_str(env, "LOG_LEVEL", default="INFO"),
        format=_read_str(env, "LOG_FORMAT", default="text"),
    )
    proxy = ProxySettings(public_base_url=_read_optional_str(env, "PUBLIC_BASE_URL"))
    return Settings(
        app=app,
        postgres=postgres,
        qdrant=qdrant,
        openai=openai,
        chat=chat,
        embedding=embedding,
        reranker=reranker,
        ingest=ingest,
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


def _read_optional_path(env: Mapping[str, str], name: str) -> Optional[Path]:
    """Read an optional filesystem path from a mapping."""
    value = _read_optional_str(env, name)
    return Path(value) if value is not None else None


def _read_int(
    env: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int = 1,
) -> int:
    """Read and validate an integer value from a mapping."""
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


def _read_optional_int(
    env: Mapping[str, str],
    name: str,
    *,
    minimum: int = 1,
) -> Optional[int]:
    """Read and validate an optional integer value from a mapping."""
    raw_value = env.get(name)
    if raw_value is None or not raw_value.strip():
        return None

    try:
        value = int(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc

    if value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}")
    return value


def _read_float(
    env: Mapping[str, str],
    name: str,
    *,
    default: float,
    minimum: float = 0.0,
) -> float:
    """Read and validate a float value from a mapping."""
    raw_value = env.get(name)
    if raw_value is None or not raw_value.strip():
        value = default
    else:
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be a number") from exc

    if value <= minimum:
        raise ValueError(f"Environment variable {name} must be > {minimum:g}")
    return value


def _read_non_negative_float(
    env: Mapping[str, str],
    name: str,
    *,
    default: float,
) -> float:
    """Read and validate a float value that may be zero."""
    raw_value = env.get(name)
    if raw_value is None or not raw_value.strip():
        value = default
    else:
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"Environment variable {name} must be a number") from exc

    if value < 0.0:
        raise ValueError(f"Environment variable {name} must be >= 0")
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
