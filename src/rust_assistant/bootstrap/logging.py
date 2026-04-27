"""Centralized process-level logging configuration."""

from __future__ import annotations

import json
import logging
import logging.config
from typing import Any

from rust_assistant.bootstrap.settings import LoggingSettings

logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """Render log records as one-line JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _normalize_log_level(level: str) -> str:
    """Normalize the configured log level."""
    normalized = level.strip().upper()
    if not normalized:
        raise ValueError("LOG_LEVEL cannot be empty")
    return normalized


def _normalize_log_format(log_format: str) -> str:
    """Normalize the configured log format name."""
    normalized = log_format.strip().lower()
    if normalized not in {"text", "json"}:
        raise ValueError("LOG_FORMAT must be either 'text' or 'json'")
    return normalized


def _build_logging_config(level: str, log_format: str) -> dict[str, Any]:
    """Build a dictConfig payload for the requested log format."""
    formatter_name = _normalize_log_format(log_format)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "text": {
                "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": JsonFormatter,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": formatter_name,
                "stream": "ext://sys.stderr",
            }
        },
        "root": {
            "level": level,
            "handlers": ["default"],
        },
        "loggers": {
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
            "sqlalchemy.pool": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
            "asyncpg": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn": {
                "level": level,
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": level,
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": level,
                "handlers": ["default"],
                "propagate": False,
            },
        },
    }


def configure_logging(*, logging_settings: LoggingSettings) -> None:
    """Configure application logging for the current process."""
    level = _normalize_log_level(logging_settings.level)
    log_format = _normalize_log_format(logging_settings.format)
    logging.config.dictConfig(_build_logging_config(level, log_format))
    logger.debug("Logging configured with level=%s format=%s", level, log_format)


__all__ = ["JsonFormatter", "configure_logging"]
