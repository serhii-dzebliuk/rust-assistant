"""System endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for the liveness endpoint."""

    status: str = "ok"
