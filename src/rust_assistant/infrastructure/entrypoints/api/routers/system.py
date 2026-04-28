"""System endpoints for the serving API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from rust_assistant.infrastructure.entrypoints.api.schemas.system import HealthResponse, ReadyResponse


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness endpoint placeholder until the real implementation is built."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Not implemented",
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Readiness endpoint placeholder until the real implementation is built."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Not implemented",
    )
