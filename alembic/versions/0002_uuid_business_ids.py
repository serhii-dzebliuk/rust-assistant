"""Switch canonical documents and chunks to UUID business identifiers."""

from __future__ import annotations

from uuid import UUID, uuid5

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_uuid_business_ids"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None

DOCUMENT_ID_NAMESPACE = UUID("3d73b6b9-a8cb-4da8-b4b0-5ecdb09b3836")
CHUNK_ID_NAMESPACE = UUID("82f00626-72d1-4ee7-90a5-c281a4f3f4ec")


def build_document_uuid(source_path: str) -> UUID:
    """Build the stable business UUID for one document row."""

    return uuid5(DOCUMENT_ID_NAMESPACE, source_path)


def build_chunk_uuid(source_path: str, chunk_index: int) -> UUID:
    """Build the stable business UUID for one chunk row."""

    document_id = build_document_uuid(source_path)
    return uuid5(CHUNK_ID_NAMESPACE, f"{document_id}:{chunk_index}")


def upgrade() -> None:
    connection = op.get_bind()

    op.alter_column("documents", "id", new_column_name="pk")
    op.add_column("documents", sa.Column("id", postgresql.UUID(as_uuid=True), nullable=True))

    document_rows = connection.execute(
        sa.text("SELECT pk, source_path FROM documents")
    ).mappings()
    for row in document_rows:
        document_id = build_document_uuid(row["source_path"])
        connection.execute(
            sa.text("UPDATE documents SET id = :id WHERE pk = :pk"),
            {"id": str(document_id), "pk": row["pk"]},
        )

    op.alter_column(
        "documents",
        "id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.create_unique_constraint(op.f("uq_documents_id"), "documents", ["id"])

    op.drop_constraint(op.f("uq_chunks_document_id_chunk_index"), "chunks", type_="unique")
    op.drop_constraint(op.f("fk_chunks_document_id_documents"), "chunks", type_="foreignkey")
    op.alter_column("chunks", "id", new_column_name="pk")
    op.alter_column("chunks", "document_id", new_column_name="document_pk")
    op.add_column("chunks", sa.Column("id", postgresql.UUID(as_uuid=True), nullable=True))

    chunk_rows = connection.execute(
        sa.text(
            """
            SELECT chunks.pk, documents.source_path, chunks.chunk_index
            FROM chunks
            JOIN documents ON documents.pk = chunks.document_pk
            """
        )
    ).mappings()
    for row in chunk_rows:
        chunk_id = build_chunk_uuid(row["source_path"], row["chunk_index"])
        connection.execute(
            sa.text("UPDATE chunks SET id = :id WHERE pk = :pk"),
            {"id": str(chunk_id), "pk": row["pk"]},
        )

    op.alter_column(
        "chunks",
        "id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.create_unique_constraint(op.f("uq_chunks_id"), "chunks", ["id"])
    op.create_foreign_key(
        op.f("fk_chunks_document_pk_documents"),
        "chunks",
        "documents",
        ["document_pk"],
        ["pk"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "uq_chunks_document_pk_chunk_index",
        "chunks",
        ["document_pk", "chunk_index"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_chunks_document_pk_chunk_index", "chunks", type_="unique")
    op.drop_constraint(op.f("fk_chunks_document_pk_documents"), "chunks", type_="foreignkey")
    op.drop_constraint(op.f("uq_chunks_id"), "chunks", type_="unique")
    op.drop_column("chunks", "id")
    op.alter_column("chunks", "document_pk", new_column_name="document_id")
    op.alter_column("chunks", "pk", new_column_name="id")

    op.drop_constraint(op.f("uq_documents_id"), "documents", type_="unique")
    op.drop_column("documents", "id")
    op.alter_column("documents", "pk", new_column_name="id")
    op.create_foreign_key(
        op.f("fk_chunks_document_id_documents"),
        "chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        op.f("uq_chunks_document_id_chunk_index"),
        "chunks",
        ["document_id", "chunk_index"],
    )
