"""SQLAlchemy ORM models for canonical PostgreSQL storage."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, Identity, Integer, Text, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rust_assistant.infrastructure.outbound.sqlalchemy.base import Base


class DocumentRecord(Base):
    """Canonical parsed document stored in PostgreSQL."""

    __tablename__ = "documents"

    pk: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False, unique=True)
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


class ChunkRecord(Base):
    """Canonical retrieval chunk stored in PostgreSQL."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_pk", "chunk_index", name="uq_chunks_document_pk_chunk_index"),
    )

    pk: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False, unique=True)
    document_pk: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.pk", ondelete="CASCADE"),
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
