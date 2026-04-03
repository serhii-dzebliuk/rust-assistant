"""Run the FastAPI backend with environment-configurable host and port."""

from __future__ import annotations

import os

import uvicorn


def _as_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    """Start the API process."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = _as_bool(os.getenv("RELOAD"))

    uvicorn.run(
        "rustrag.serving.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
