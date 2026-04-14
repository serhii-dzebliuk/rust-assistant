"""System endpoints for the serving API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rust_assistant.schemas.system import HealthResponse, ReadyResponse
from rust_assistant.services.readiness_service import ReadinessService

from ..deps import get_readiness_service


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(
    readiness_service: ReadinessService = Depends(get_readiness_service),
) -> HealthResponse:
    """Liveness endpoint for the API process."""
    health_status = readiness_service.health()
    return HealthResponse(status=health_status.status, mode=health_status.mode)


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    readiness_service: ReadinessService = Depends(get_readiness_service),
) -> ReadyResponse:
    """Readiness endpoint backed by the readiness application service."""
    readiness = await readiness_service.readiness()
    return ReadyResponse(
        status=readiness.status,
        ready=readiness.ready,
        mode=readiness.mode,
        dependencies=readiness.dependencies,
    )
