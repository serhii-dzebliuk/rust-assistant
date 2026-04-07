"""System endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for the liveness endpoint."""

    status: str = "ok"
    # TODO: Remove the stub default once endpoints always set the runtime mode.
    mode: str = "stub"


class ReadyResponse(BaseModel):
    """Response for the readiness endpoint."""

    status: str = "ready"
    # TODO: Replace the optimistic stub default with real readiness derived
    # from dependency checks.
    ready: bool = True
    # TODO: Remove the stub default once endpoints always set the runtime mode.
    mode: str = "stub"
    dependencies: dict[str, str] = Field(default_factory=dict)


