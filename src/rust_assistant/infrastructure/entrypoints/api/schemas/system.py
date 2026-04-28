"""System endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for the liveness endpoint."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Response for the readiness endpoint."""

    status: str = "ready"
    ready: bool = True

