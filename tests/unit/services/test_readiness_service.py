import asyncio

import pytest

from rust_assistant.services.readiness_service import ReadinessService

pytestmark = pytest.mark.unit


async def _probe_ready() -> str:
    return "ready"


async def _probe_not_ready() -> str:
    return "not_ready"


def test_readiness_service_uses_async_dependency_probes():
    service = ReadinessService(
        mode="runtime",
        dependencies={"postgres": "not_configured", "qdrant": "not_configured"},
        dependency_probes={"postgres": _probe_ready, "qdrant": _probe_not_ready},
    )

    readiness = asyncio.run(service.readiness())

    assert readiness.mode == "runtime"
    assert readiness.ready is False
    assert readiness.status == "not_ready"
    assert readiness.dependencies == {"postgres": "ready", "qdrant": "not_ready"}
