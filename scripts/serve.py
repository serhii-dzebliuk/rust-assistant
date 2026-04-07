"""Run the FastAPI backend with centralized application settings."""

from __future__ import annotations

import logging

import uvicorn

from rust_assistant.core.config import get_settings
from rust_assistant.core.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the API process."""
    settings = get_settings()
    configure_logging(logging_settings=settings.logging)
    logger.info(
        "Starting API server on %s:%s reload=%s",
        settings.app.host,
        settings.app.port,
        settings.app.reload,
    )

    uvicorn.run(
        "rust_assistant.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload,
    )


if __name__ == "__main__":
    main()
