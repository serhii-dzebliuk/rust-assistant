"""Document chunking policy functions."""

from __future__ import annotations

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.policies.chunk_building import build_fallback_chunk
from rust_assistant.domain.policies.chunk_filtering import filter_low_value_chunks
from rust_assistant.domain.policies.chunk_merging import merge_small_chunks, reindex_chunks
from rust_assistant.domain.policies.chunk_sections import (
    build_document_sections,
    resolve_chunk_limit,
)
from rust_assistant.domain.policies.chunk_span_splitting import (
    build_block_spans,
    split_overlong_chunks,
    split_section_into_chunks,
)


def chunk_document(
    document: Document,
    *,
    max_chunk_chars: int = 1400,
    min_chunk_chars: int = 180,
) -> list[Chunk]:
    """
    Chunk one cleaned document into retrieval-ready chunks.

    Args:
        document: Cleaned document with structured blocks.
        max_chunk_chars: Default maximum chunk size in characters.
        min_chunk_chars: Minimum chunk size targeted by safe adjacent merges.

    Returns:
        Ordered list of chunks for the document.
    """

    if not document.structured_blocks:
        return [build_fallback_chunk(document)]

    sections = build_document_sections(document)
    if not sections:
        return [build_fallback_chunk(document)]

    effective_chunk_limit = resolve_chunk_limit(document.crate, max_chunk_chars)
    rendered_blocks, block_spans = build_block_spans(document.structured_blocks, document.text)

    chunks: list[Chunk] = []
    chunk_index = 0
    for section in sections:
        section_chunks = split_section_into_chunks(
            document,
            section,
            rendered_blocks,
            block_spans,
            max_chunk_chars=effective_chunk_limit,
            start_chunk_index=chunk_index,
        )
        chunks.extend(section_chunks)
        chunk_index += len(section_chunks)

    if not chunks:
        return [build_fallback_chunk(document)]

    merged_chunks = merge_small_chunks(
        document,
        chunks,
        max_chunk_chars=effective_chunk_limit,
        min_chunk_chars=min_chunk_chars,
    )
    bounded_chunks = split_overlong_chunks(
        document,
        merged_chunks,
        max_chunk_chars=effective_chunk_limit,
    )
    filtered_chunks = filter_low_value_chunks(
        bounded_chunks,
        min_chunk_chars=min_chunk_chars,
    )
    return reindex_chunks(document, filtered_chunks or bounded_chunks)


def chunk_documents(
    documents: list[Document],
    *,
    max_chunk_chars: int = 1400,
    min_chunk_chars: int = 180,
) -> list[Chunk]:
    """
    Chunk cleaned documents in memory.

    Args:
        documents: Cleaned and deduplicated documents.
        max_chunk_chars: Default maximum chunk size in characters.
        min_chunk_chars: Minimum chunk size targeted by safe adjacent merges.

    Returns:
        Ordered list of generated chunks.
    """

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_chunk_chars=max_chunk_chars,
                min_chunk_chars=min_chunk_chars,
            )
        )
    return chunks
