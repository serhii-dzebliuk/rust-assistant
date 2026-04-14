"""Shared API dependencies for the serving layer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Optional

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rust_assistant.core.db import get_db_session_context
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


async def get_db_session(request: Request) -> AsyncIterator[Optional[AsyncSession]]:
    """Yield a request-scoped async database session when configured."""
    session_factory = getattr(request.app.state, "db_session_factory", None)
    async for session in get_db_session_context(session_factory):
        yield session


def get_readiness_service(request: Request) -> ReadinessService:
    """Resolve the readiness service for the current request."""
    service = getattr(request.app.state, "readiness_service", None)
    if service is not None:
        return service

    return ReadinessService(
        mode=_get_api_mode(request),
        dependencies=_get_dependency_statuses(request),
    )


def get_search_service(
    request: Request,
    db_session: Optional[AsyncSession] = Depends(get_db_session),
) -> SearchService:
    """Resolve the search service for the current request."""
    retriever = getattr(request.app.state, "retriever", None)
    return SearchService(
        mode=_get_api_mode(request),
        retriever=retriever,
        session=db_session,
    )


def get_chat_service(
    request: Request,
    db_session: Optional[AsyncSession] = Depends(get_db_session),
) -> ChatService:
    """Resolve the chat service for the current request."""
    qa_pipeline = getattr(request.app.state, "qa_pipeline", None)
    return ChatService(
        mode=_get_api_mode(request),
        dependencies=_get_dependency_statuses(request),
        qa_pipeline=qa_pipeline,
        session=db_session,
    )
