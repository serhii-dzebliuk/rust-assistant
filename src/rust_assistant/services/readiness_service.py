"""Application service for liveness and readiness checks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Optional


READY_DEPENDENCY_STATUSES = frozenset({"ok", "ready", "connected", "not_configured"})
DependencyProbe = Callable[[], Awaitable[str]]


@dataclass(slots=True, frozen=True)
class HealthStatus:
    """Result of a liveness check for the serving process."""

    status: str
    mode: str


@dataclass(slots=True, frozen=True)
class ReadinessStatus:
    """Result of a readiness check for the serving process."""

    status: str
    ready: bool
    mode: str
    dependencies: dict[str, str]


class ReadinessService:
    """Encapsulate health and readiness decisions for the serving runtime."""

    def __init__(
        self,
        *,
        mode: str = "stub",
        dependencies: Optional[Mapping[str, str]] = None,
        dependency_probes: Optional[Mapping[str, DependencyProbe]] = None,
    ) -> None:
        self._mode = mode
        self._dependencies = dict(dependencies or {})
        self._dependency_probes = dict(dependency_probes or {})

    def health(self) -> HealthStatus:
        """Report whether the HTTP process itself is alive."""
        return HealthStatus(status="ok", mode=self._mode)

    async def readiness(self) -> ReadinessStatus:
        """Report whether the serving process is ready to accept traffic."""
        dependencies = dict(self._dependencies)
        for name, probe in self._dependency_probes.items():
            try:
                dependencies[name] = await probe()
            except Exception:
                dependencies[name] = "not_ready"

        ready = all(self._dependency_is_ready(status) for status in dependencies.values())
        return ReadinessStatus(
            status="ready" if ready else "not_ready",
            ready=ready,
            mode=self._mode,
            dependencies=dependencies,
        )

    @staticmethod
    def _dependency_is_ready(status: str) -> bool:
        """Interpret dependency status values conservatively for now."""
        return status in READY_DEPENDENCY_STATUSES
