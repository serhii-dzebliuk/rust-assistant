"""Chunk splitting helpers that operate on block spans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from rust_assistant.domain.entities.chunks import Chunk
from rust_assistant.domain.entities.documents import Document
from rust_assistant.domain.errors import ChunkingError
from rust_assistant.domain.policies.chunk_boundaries import (
    is_single_fenced_code_block,
    split_rendered_lines,
    split_text_by_boundaries,
)
from rust_assistant.domain.policies.chunk_building import build_chunk
from rust_assistant.domain.policies.chunk_sections import DocumentSection
from rust_assistant.domain.policies.text_rendering import blocks_to_text
from rust_assistant.domain.value_objects.structured_blocks import (
    BlockType,
    StructuredBlock,
)


def build_block_spans(
    blocks: Sequence[StructuredBlock],
    document_text: str,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Render blocks and compute their character spans in document text."""

    rendered_blocks = [blocks_to_text([block]) for block in blocks]
    spans: list[tuple[int, int]] = []
    position = 0
    for rendered in rendered_blocks:
        start = document_text.find(rendered, position)
        if start < 0:
            start = position
        end = start + len(rendered)
        spans.append((start, end))
        position = end
    return rendered_blocks, spans


def split_section_into_chunks(
    document: Document,
    section: DocumentSection,
    rendered_blocks: list[str],
    block_spans: list[tuple[int, int]],
    *,
    max_chunk_chars: int,
    start_chunk_index: int,
) -> list[Chunk]:
    """Split one logical section into one or more chunks."""

    section_indexes = list(section.block_indexes)
    if not section_indexes:
        return []

    first_index = section_indexes[0]
    heading_index: Optional[int] = None
    if document.structured_blocks[first_index].block_type == BlockType.HEADING:
        heading_index = first_index
        content_indexes = section_indexes[1:]
    else:
        content_indexes = section_indexes

    if not content_indexes:
        return []

    chunks: list[Chunk] = []
    current_chunk_indexes = [heading_index] if heading_index is not None else []
    current_content_indexes: list[int] = []
    current_length = _joined_length(current_chunk_indexes, rendered_blocks)

    for index in content_indexes:
        block_length = len(rendered_blocks[index])
        block = document.structured_blocks[index]
        separator = 2 if current_chunk_indexes else 0
        would_overflow_empty_section = (
            not current_content_indexes
            and current_chunk_indexes
            and current_length + separator + block_length > max_chunk_chars
        )

        if not current_content_indexes and (
            block_length > max_chunk_chars or would_overflow_empty_section
        ):
            oversized_chunks = _split_oversized_block(
                document,
                section,
                heading_index=heading_index if not chunks else None,
                block_index=index,
                block=block,
                block_spans=block_spans,
                max_chunk_chars=max_chunk_chars,
                chunk_index_offset=start_chunk_index + len(chunks),
            )
            if oversized_chunks:
                chunks.extend(oversized_chunks)
                current_chunk_indexes = []
                current_content_indexes = []
                current_length = 0
                continue

        would_overflow = (
            current_content_indexes
            and current_length + separator + block_length > max_chunk_chars
        )
        if would_overflow:
            chunks.append(
                _build_section_chunk(
                    document,
                    chunk_indexes=current_chunk_indexes,
                    section=section,
                    block_spans=block_spans,
                    chunk_index=start_chunk_index + len(chunks),
                )
            )
            current_chunk_indexes = []
            current_content_indexes = []
            current_length = _joined_length(current_chunk_indexes, rendered_blocks)
            if block_length > max_chunk_chars:
                oversized_chunks = _split_oversized_block(
                    document,
                    section,
                    heading_index=None,
                    block_index=index,
                    block=block,
                    block_spans=block_spans,
                    max_chunk_chars=max_chunk_chars,
                    chunk_index_offset=start_chunk_index + len(chunks),
                )
                if oversized_chunks:
                    chunks.extend(oversized_chunks)
                    current_length = 0
                    continue

            separator = 2 if current_chunk_indexes else 0

        current_chunk_indexes.append(index)
        current_content_indexes.append(index)
        current_length += separator + block_length

    if current_content_indexes:
        chunks.append(
            _build_section_chunk(
                document,
                chunk_indexes=current_chunk_indexes,
                section=section,
                block_spans=block_spans,
                chunk_index=start_chunk_index + len(chunks),
            )
        )

    return chunks


def split_overlong_chunks(
    document: Document,
    chunks: list[Chunk],
    *,
    max_chunk_chars: int,
) -> list[Chunk]:
    """Enforce a final character cap using safe text boundaries."""

    bounded_chunks: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) <= max_chunk_chars:
            bounded_chunks.append(chunk)
            continue
        if is_single_fenced_code_block(chunk.text):
            raise ChunkingError(
                "Cannot safely split a single fenced code block over "
                f"{max_chunk_chars} characters in {chunk.source_path}"
            )

        relative_groups = split_text_by_boundaries(chunk.text, max_chunk_chars)
        relative_start = 0
        for group_text, consumed_length in relative_groups:
            start_offset = chunk.start_offset + relative_start
            end_offset = start_offset + len(group_text)
            relative_start += consumed_length
            split_text = document.text[start_offset:end_offset]
            if not split_text.strip():
                continue
            bounded_chunks.append(
                build_chunk(
                    document,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    chunk_index=chunk.chunk_index,
                    section_path=chunk.section_path or (document.title,),
                    anchor=chunk.section_anchor,
                )
            )

    return bounded_chunks


def _build_section_chunk(
    document: Document,
    *,
    chunk_indexes: list[int],
    section: DocumentSection,
    block_spans: list[tuple[int, int]],
    chunk_index: int,
) -> Chunk:
    start_index = chunk_indexes[0]
    end_index = chunk_indexes[-1]
    start_offset = block_spans[start_index][0]
    end_offset = block_spans[end_index][1]
    return build_chunk(
        document,
        start_offset=start_offset,
        end_offset=end_offset,
        chunk_index=chunk_index,
        section_path=section.section_path,
        anchor=section.anchor,
    )


def _joined_length(block_indexes: list[int], rendered_blocks: list[str]) -> int:
    if not block_indexes:
        return 0
    total = sum(len(rendered_blocks[index]) for index in block_indexes)
    total += 2 * (len(block_indexes) - 1)
    return total


def _split_oversized_block(
    document: Document,
    section: DocumentSection,
    *,
    heading_index: Optional[int],
    block_index: int,
    block: StructuredBlock,
    block_spans: list[tuple[int, int]],
    max_chunk_chars: int,
    chunk_index_offset: int,
) -> list[Chunk]:
    block_start, block_end = block_spans[block_index]
    rendered_block_text = document.text[block_start:block_end]
    heading_start = block_spans[heading_index][0] if heading_index is not None else None
    heading_overhead = 0
    if heading_start is not None:
        heading_overhead = block_start - heading_start

    target_block_chars = max(max_chunk_chars - heading_overhead, 1)
    if block.block_type == BlockType.CODE_BLOCK:
        text_groups = split_rendered_lines(rendered_block_text, target_block_chars)
    else:
        text_groups = split_text_by_boundaries(rendered_block_text, target_block_chars)

    if len(text_groups) <= 1:
        return []

    chunks: list[Chunk] = []
    content_offset = 0
    for group_index, (group_text, consumed_length) in enumerate(text_groups):
        start_char = block_start + content_offset
        end_char = start_char + len(group_text)
        if group_index == 0 and heading_start is not None:
            start_char = heading_start
        content_offset += consumed_length

        chunks.append(
            build_chunk(
                document,
                start_offset=start_char,
                end_offset=end_char,
                chunk_index=chunk_index_offset + group_index,
                section_path=section.section_path,
                anchor=section.anchor,
            )
        )

    return chunks
