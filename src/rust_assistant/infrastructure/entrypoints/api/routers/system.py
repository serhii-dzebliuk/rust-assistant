"""System endpoints for the serving API."""

from __future__ import annotations

from fastapi import APIRouter

from rust_assistant.infrastructure.entrypoints.api.schemas.system import HealthResponse, ReadyResponse


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API process is alive."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Report that the API process is ready to receive requests."""
    return ReadyResponse()
