"""Document ORM model."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, Identity, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rust_assistant.core.db.base import Base

if TYPE_CHECKING:
    from .chunk import ChunkRecord


class DocumentRecord(Base):
    """Canonical parsed document stored in PostgreSQL."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    crate: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_content: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    item_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rust_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    item_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    chunks: Mapped[list["ChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


__all__ = ["DocumentRecord"]
