"""Mapping helpers between ORM records and lean domain models."""

from __future__ import annotations

from typing import Any

from rust_assistant.application.dto.chunk_context import ChunkContext
from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.enums import Crate, ItemType
from rust_assistant.domain.value_objects.identifiers import (
    ChunkId,
    DocumentId,
    build_chunk_id,
    build_document_id,
)
from rust_assistant.domain.value_objects.structured_blocks import (
    BlockType,
    StructuredBlock,
)
from rust_assistant.infrastructure.outbound.sqlalchemy.models import ChunkRecord, DocumentRecord


def map_document_to_domain(document: DocumentRecord) -> Document:
    """Convert a document ORM record into a domain entity."""

    domain_document = Document(
        source_path=document.source_path,
        title=document.title,
        text=document.text_content,
        crate=Crate(document.crate),
        url=document.url,
        item_path=document.item_path,
        item_type=ItemType(document.item_type) if document.item_type else None,
        rust_version=document.rust_version,
        structured_blocks=tuple(
            _map_structured_block_from_payload(block_payload)
            for block_payload in document.parsed_content
        ),
    )
    _validate_document_identity(document, domain_document)
    return domain_document


def map_document_from_domain(document: Document) -> DocumentRecord:
    """Create a new ORM document record from a domain entity."""

    record = DocumentRecord(id=document.id, source_path=document.source_path)
    return apply_document_to_record(record, document)


def apply_document_to_record(record: DocumentRecord, document: Document) -> DocumentRecord:
    """Copy a domain document into an ORM record instance."""

    record.id = document.id
    record.source_path = document.source_path
    record.crate = document.crate.value
    record.title = document.title
    record.text_content = document.text
    record.parsed_content = [
        _map_structured_block_to_payload(block)
        for block in document.structured_blocks
    ]
    record.url = document.url
    record.item_path = document.item_path
    record.rust_version = document.rust_version
    record.item_type = document.item_type.value if document.item_type else None
    return record


def map_chunk_to_domain(chunk: ChunkRecord) -> Chunk:
    """Convert a chunk ORM record into a domain entity."""

    if chunk.start_offset is None or chunk.end_offset is None:
        raise ValueError("Chunk offsets must be present when mapping ORM chunks to domain")

    document = chunk.document
    domain_chunk = Chunk(
        source_path=document.source_path,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        crate=Crate(document.crate),
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        section_path=tuple(chunk.section_path or ()),
        section_anchor=chunk.section_anchor,
        item_path=document.item_path,
        item_type=ItemType(document.item_type) if document.item_type else None,
        rust_version=document.rust_version,
        url=document.url,
        token_count=chunk.token_count,
        text_hash=chunk.hash,
    )
    _validate_chunk_identity(chunk, domain_chunk)
    return domain_chunk


def map_chunk_from_domain(chunk: Chunk, document_pk: int) -> ChunkRecord:
    """Create a new ORM chunk record from a domain entity."""

    record = ChunkRecord(id=chunk.id, document_pk=document_pk, chunk_index=chunk.chunk_index)
    return apply_chunk_to_record(record, chunk, document_pk)


def apply_chunk_to_record(
    record: ChunkRecord,
    chunk: Chunk,
    document_pk: int,
) -> ChunkRecord:
    """Copy a domain chunk into an ORM record instance."""

    record.id = chunk.id
    record.document_pk = document_pk
    record.chunk_index = chunk.chunk_index
    record.text = chunk.text
    record.hash = chunk.text_hash
    record.token_count = chunk.token_count
    record.section_title = chunk.section_title
    record.section_anchor = chunk.section_anchor
    record.section_path = list(chunk.section_path) if chunk.section_path else None
    record.start_offset = chunk.start_offset
    record.end_offset = chunk.end_offset
    return record


def map_chunk_context_from_record(chunk: ChunkRecord) -> ChunkContext:
    """Convert an ORM chunk record into a retrieval-oriented read DTO."""

    document = chunk.document
    return ChunkContext(
        chunk_id=ChunkId(chunk.id),
        document_id=DocumentId(document.id),
        text=chunk.text,
        title=document.title,
        source_path=document.source_path,
        url=document.url,
        section_title=chunk.section_title,
        section_path=tuple(chunk.section_path or ()),
        section_anchor=chunk.section_anchor,
        item_path=document.item_path,
        crate=Crate(document.crate),
        item_type=ItemType(document.item_type) if document.item_type else None,
        rust_version=document.rust_version,
        chunk_index=chunk.chunk_index,
    )


def _validate_document_identity(record: DocumentRecord, document: Document) -> None:
    """Ensure persisted and derived document identities stay aligned."""

    expected_id = build_document_id(record.source_path)
    if record.id != expected_id or record.id != document.id:
        raise ValueError(
            "Document identity mismatch between persisted UUID and derived domain identity"
        )


def _validate_chunk_identity(record: ChunkRecord, chunk: Chunk) -> None:
    """Ensure persisted and derived chunk identities stay aligned."""

    document = record.document
    expected_document_id = build_document_id(document.source_path)
    expected_chunk_id = build_chunk_id(expected_document_id, record.chunk_index)
    if document.id != expected_document_id or chunk.document_id != expected_document_id:
        raise ValueError(
            "Chunk parent document identity mismatch between persisted UUID and domain identity"
        )
    if record.id != expected_chunk_id or chunk.id != expected_chunk_id:
        raise ValueError(
            "Chunk identity mismatch between persisted UUID and derived domain identity"
        )


def _map_structured_block_from_payload(block_payload: dict[str, Any]) -> StructuredBlock:
    """Build a structured block from persisted JSON payload."""

    return StructuredBlock(
        block_type=BlockType(block_payload["block_type"]),
        text=block_payload["text"],
        html_tag=block_payload["html_tag"],
        heading_level=block_payload.get("heading_level"),
        list_depth=block_payload.get("list_depth"),
        code_language=block_payload.get("code_language"),
        anchor=block_payload.get("anchor"),
        section_path=tuple(block_payload.get("section_path") or ()),
    )


def _map_structured_block_to_payload(block: StructuredBlock) -> dict[str, Any]:
    """Convert a structured block into persisted JSON payload."""

    return {
        "block_type": block.block_type.value,
        "text": block.text,
        "html_tag": block.html_tag,
        "heading_level": block.heading_level,
        "list_depth": block.list_depth,
        "code_language": block.code_language,
        "anchor": block.anchor,
        "section_path": list(block.section_path),
    }
