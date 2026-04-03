"""Routers exposed by the serving API."""

from .chat import router as chat_router
from .search import router as search_router
from .system import router as system_router

__all__ = ["chat_router", "search_router", "system_router"]
