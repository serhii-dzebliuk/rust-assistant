"""Application service for liveness and readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


READY_DEPENDENCY_STATUSES = frozenset({"ok", "ready", "connected", "not_configured"})


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
        dependencies: Mapping[str, str] | None = None,
    ) -> None:
        self._mode = mode
        self._dependencies = dict(dependencies or {})

    def health(self) -> HealthStatus:
        """Report whether the HTTP process itself is alive."""
        return HealthStatus(status="ok", mode=self._mode)

    def readiness(self) -> ReadinessStatus:
        """Report whether the serving process is ready to accept traffic."""
        dependencies = dict(self._dependencies)
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
