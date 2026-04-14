"""Chunk ORM model."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Identity, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rust_assistant.core.db.base import Base

if TYPE_CHECKING:
    from .document import DocumentRecord


class ChunkRecord(Base):
    """Canonical retrieval chunk stored in PostgreSQL."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_id_chunk_index"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    hash: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    section_anchor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    section_path: Mapped[Optional[list[str]]] = mapped_column(JSONB, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_offset: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end_offset: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    document: Mapped["DocumentRecord"] = relationship(back_populates="chunks")


__all__ = ["ChunkRecord"]
