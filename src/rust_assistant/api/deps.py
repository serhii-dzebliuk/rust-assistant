"""Shared API dependencies for the serving layer."""

from __future__ import annotations

from fastapi import Request

from rust_assistant.services import ChatService, ReadinessService, SearchService


def _get_api_mode(request: Request) -> str:
    """Read the current serving mode from app state."""
    return getattr(request.app.state, "api_mode", "stub")


def _get_dependency_statuses(request: Request) -> dict[str, str]:
    """Read dependency statuses from app state."""
    return dict(
        getattr(
            request.app.state,
            "dependencies",
            {"postgres": "not_configured", "qdrant": "not_configured"},
        )
    )


def get_readiness_service(request: Request) -> ReadinessService:
    """Resolve the readiness service for the current request."""
    service = getattr(request.app.state, "readiness_service", None)
    if service is not None:
        return service

    return ReadinessService(
        mode=_get_api_mode(request),
        dependencies=_get_dependency_statuses(request),
    )


def get_search_service(request: Request) -> SearchService:
    """Resolve the search service for the current request."""
    service = getattr(request.app.state, "search_service", None)
    if service is not None:
        return service

    return SearchService(mode=_get_api_mode(request))


def get_chat_service(request: Request) -> ChatService:
    """Resolve the chat service for the current request."""
    service = getattr(request.app.state, "chat_service", None)
    if service is not None:
        return service

    return ChatService(
        mode=_get_api_mode(request),
        dependencies=_get_dependency_statuses(request),
    )
