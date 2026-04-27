"""Domain exceptions."""


class DomainError(ValueError):
    """Base exception for domain-level failures."""


class InvalidChunkTextError(DomainError):
    """Raised when chunk text is empty or invalid."""


class ChunkingError(DomainError):
    """Raised when text cannot be chunked safely."""
