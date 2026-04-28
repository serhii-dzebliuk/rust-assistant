"""Adapter-local configuration for SQLAlchemy runtime wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True, frozen=True)
class SqlAlchemyConfig:
    """Concrete runtime configuration for the SQLAlchemy adapter."""

    url: Optional[str]
    echo: bool
    pool_size: int
    max_overflow: int
