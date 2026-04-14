"""
Chunk cleaned documents into retrieval-ready text spans.

This module implements ingest stage 1.6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from rust_assistant.ingest.entities import BlockType, Chunk, ChunkMetadata, Document, StructuredBlock
from rust_assistant.schemas.enums import Crate

from .parsing.core import blocks_to_text

logger = logging.getLogger(__name__)


RUSTDOC_CRATES = {
    Crate.STD,
    Crate.CORE,
    Crate.ALLOC,
    Crate.PROC_MACRO,
    Crate.TEST,
}


@dataclass(slots=True)
class SectionSpan:
    """
    Contiguous document block span representing one logical section.

    Attributes:
        block_indexes: Ordered indexes of blocks belonging to the section.
        section_path: Heading path active for the section.
        anchor: Section anchor used for chunk metadata.
    """

    block_indexes: list[int]
    section_path: list[str]
    anchor: Optional[str]


class BaseChunkStrategy:
    """
    Base strategy for chunking a document from structured blocks.

    The strategy first groups blocks into logical sections, then splits
    oversized sections by block boundaries while keeping code blocks intact.
    """

    max_chunk_chars = 1400

    def build_sections(self, doc: Document) -> list[SectionSpan]:
        """
        Split a document into contiguous logical sections.

        Args:
            doc: Cleaned document with structured blocks.

        Returns:
            Ordered list of section spans.
        """
        if not doc.structured_blocks:
            return []

        sections: list[SectionSpan] = []
        current_indexes: list[int] = []

        for index, block in enumerate(doc.structured_blocks):
            if self._starts_new_section(block):
                if current_indexes:
                    sections.append(self._build_section(doc, current_indexes))
                current_indexes = [index]
                continue

            if not current_indexes:
                current_indexes = [index]
            else:
                current_indexes.append(index)

        if current_indexes:
            sections.append(self._build_section(doc, current_indexes))

        meaningful_sections = [
            section
            for section in sections
            if any(
                doc.structured_blocks[index].block_type != BlockType.HEADING
                for index in section.block_indexes
            )
        ]
        return meaningful_sections or sections

    def _starts_new_section(self, block: StructuredBlock) -> bool:
        """
        Check whether a block should start a new section.

        Args:
            block: Candidate structured block.

        Returns:
            `True` when the block marks a new section boundary.
        """
        return block.block_type == BlockType.HEADING

    def _build_section(self, doc: Document, block_indexes: list[int]) -> SectionSpan:
        """
        Build a section span from contiguous block indexes.

        Args:
            doc: Parent document.
            block_indexes: Ordered block indexes belonging to the section.

        Returns:
            Section span with section metadata copied from the leading block.
        """
        lead_block = doc.structured_blocks[block_indexes[0]]
        section_path = list(lead_block.section_path)
        if not section_path:
            section_path = [doc.title]

        anchor = lead_block.anchor
        if anchor is None:
            for index in block_indexes:
                candidate = doc.structured_blocks[index].anchor
                if candidate is not None:
                    anchor = candidate
                    break

        return SectionSpan(
            block_indexes=block_indexes,
            section_path=section_path,
            anchor=anchor,
        )


class RustdocChunkStrategy(BaseChunkStrategy):
    """
    Chunking strategy for rustdoc-style API documentation.

    Rustdoc pages naturally expose many semantically meaningful headings such as
    `Examples`, `Safety`, `Panics`, method signatures, and trait impl sections.
    Every heading is treated as a section boundary.
    """

    max_chunk_chars = 1350


class BookChunkStrategy(BaseChunkStrategy):
    """
    Chunking strategy for narrative mdBook-style documentation pages.
    """

    max_chunk_chars = 1500


class CargoChunkStrategy(BaseChunkStrategy):
    """
    Chunking strategy for Cargo book pages and command/reference pages.
    """

    max_chunk_chars = 1450


class ReferenceChunkStrategy(BaseChunkStrategy):
    """
    Chunking strategy for Rust Reference pages.
    """

    max_chunk_chars = 1300


class DocumentChunker:
    """
    Convert cleaned documents into retrieval-ready chunks.

    Args:
        max_chunk_chars: Default maximum chunk size in characters.
    """

    def __init__(self, max_chunk_chars: int = 1400, min_chunk_chars: int = 180):
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """
        Chunk one document into retrieval-ready chunks.

        Args:
            doc: Cleaned document with structured blocks.

        Returns:
            Ordered list of chunks for the document.
        """
        if not doc.structured_blocks:
            return [self._fallback_chunk(doc)]

        strategy = self._strategy_for(doc.metadata.crate)
        sections = strategy.build_sections(doc)
        if not sections:
            return [self._fallback_chunk(doc)]

        rendered_blocks, block_spans = self._build_block_spans(doc.structured_blocks)
        chunks: list[Chunk] = []
        chunk_index = 0
        for section in sections:
            section_chunks = self._chunk_section(
                doc=doc,
                section=section,
                rendered_blocks=rendered_blocks,
                block_spans=block_spans,
                max_chunk_chars=min(strategy.max_chunk_chars, self.max_chunk_chars),
                start_chunk_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        if not chunks:
            return [self._fallback_chunk(doc)]

        merged_chunks = self._merge_small_chunks(
            doc=doc,
            chunks=chunks,
            max_chunk_chars=min(strategy.max_chunk_chars, self.max_chunk_chars),
        )
        return self._reindex_chunks(doc, merged_chunks)

    def _strategy_for(self, crate: Crate) -> BaseChunkStrategy:
        """
        Select chunking strategy for a crate.

        Args:
            crate: Document crate identifier.

        Returns:
            Strategy instance tuned for the crate layout.
        """
        if crate in RUSTDOC_CRATES:
            return RustdocChunkStrategy()
        if crate == Crate.BOOK:
            return BookChunkStrategy()
        if crate == Crate.CARGO:
            return CargoChunkStrategy()
        if crate == Crate.REFERENCE:
            return ReferenceChunkStrategy()
        return BaseChunkStrategy()

    def _build_block_spans(
        self,
        blocks: list[StructuredBlock],
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """
        Render blocks and compute their character spans in document text.

        Args:
            blocks: Structured document blocks in source order.

        Returns:
            Tuple of rendered block text and `(start, end)` spans.
        """
        rendered_blocks = [blocks_to_text([block]) for block in blocks]
        spans: list[tuple[int, int]] = []
        position = 0
        for index, rendered in enumerate(rendered_blocks):
            start = position
            end = start + len(rendered)
            spans.append((start, end))
            position = end
            if index < len(rendered_blocks) - 1:
                position += 2
        return rendered_blocks, spans

    def _chunk_section(
        self,
        doc: Document,
        section: SectionSpan,
        rendered_blocks: list[str],
        block_spans: list[tuple[int, int]],
        max_chunk_chars: int,
        start_chunk_index: int,
    ) -> list[Chunk]:
        """
        Split one logical section into one or more chunks.

        Args:
            doc: Parent document.
            section: Logical section span.
            rendered_blocks: Rendered per-block text.
            block_spans: Character spans of rendered blocks inside `doc.text`.
            max_chunk_chars: Maximum desired chunk size.
            start_chunk_index: Chunk index offset for the document.

        Returns:
            Ordered chunks belonging to the section.
        """
        section_indexes = list(section.block_indexes)
        if not section_indexes:
            return []

        first_index = section_indexes[0]
        heading_index: Optional[int] = None
        if doc.structured_blocks[first_index].block_type == BlockType.HEADING:
            heading_index = first_index
            content_indexes = section_indexes[1:]
        else:
            content_indexes = section_indexes

        if not content_indexes:
            return []

        chunks: list[Chunk] = []
        current_chunk_indexes = [heading_index] if heading_index is not None else []
        current_content_indexes: list[int] = []
        current_length = self._joined_length(current_chunk_indexes, rendered_blocks)

        for index in content_indexes:
            block_length = len(rendered_blocks[index])
            block = doc.structured_blocks[index]

            if not current_content_indexes and block_length > max_chunk_chars:
                oversized_chunks = self._split_oversized_block(
                    doc=doc,
                    section=section,
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

            separator = 2 if current_chunk_indexes else 0
            would_overflow = (
                current_content_indexes
                and current_length + separator + block_length > max_chunk_chars
            )
            if would_overflow:
                chunks.append(
                    self._build_chunk(
                        doc=doc,
                        chunk_indexes=current_chunk_indexes,
                        section=section,
                        block_spans=block_spans,
                        chunk_index=start_chunk_index + len(chunks),
                    )
                )
                current_chunk_indexes = []
                current_content_indexes = []
                current_length = self._joined_length(current_chunk_indexes, rendered_blocks)
                if block_length > max_chunk_chars:
                    oversized_chunks = self._split_oversized_block(
                        doc=doc,
                        section=section,
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
                self._build_chunk(
                    doc=doc,
                    chunk_indexes=current_chunk_indexes,
                    section=section,
                    block_spans=block_spans,
                    chunk_index=start_chunk_index + len(chunks),
                )
            )

        return chunks

    def _joined_length(
        self,
        block_indexes: list[int],
        rendered_blocks: list[str],
    ) -> int:
        """
        Compute joined text length for a block slice.

        Args:
            block_indexes: Ordered block indexes to render together.
            rendered_blocks: Rendered per-block text cache.

        Returns:
            Character length of rendered block slice.
        """
        if not block_indexes:
            return 0
        total = sum(len(rendered_blocks[index]) for index in block_indexes)
        total += 2 * (len(block_indexes) - 1)
        return total

    def _build_chunk(
        self,
        doc: Document,
        chunk_indexes: list[int],
        section: SectionSpan,
        block_spans: list[tuple[int, int]],
        chunk_index: int,
    ) -> Chunk:
        """
        Materialize a chunk from block indexes and metadata context.

        Args:
            doc: Parent document.
            chunk_indexes: Block indexes rendered into the chunk text.
            section: Logical parent section span.
            block_spans: Character spans inside document text.
            chunk_index: Chunk ordinal inside the document.

        Returns:
            Chunk instance with filled metadata.
        """
        start_index = chunk_indexes[0]
        end_index = chunk_indexes[-1]
        start_char = block_spans[start_index][0]
        end_char = block_spans[end_index][1]
        chunk_text = doc.text[start_char:end_char]

        leaf_section = section.section_path[-1] if section.section_path else None
        metadata = ChunkMetadata(
            crate=doc.metadata.crate,
            item_path=doc.metadata.item_path,
            item_type=doc.metadata.item_type,
            rust_version=doc.metadata.rust_version,
            url=doc.metadata.url,
            section=leaf_section,
            section_path=section.section_path,
            anchor=section.anchor,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            doc_title=doc.title,
            doc_source_path=doc.source_path,
        )
        return Chunk(
            chunk_id=Chunk.generate_id(doc.doc_id, chunk_index),
            doc_id=doc.doc_id,
            text=chunk_text,
            metadata=metadata,
        )

    def _split_oversized_block(
        self,
        doc: Document,
        section: SectionSpan,
        heading_index: Optional[int],
        block_index: int,
        block: StructuredBlock,
        block_spans: list[tuple[int, int]],
        max_chunk_chars: int,
        chunk_index_offset: int,
    ) -> list[Chunk]:
        """
        Split one oversized block into multiple retrieval-sized chunks.

        Args:
            doc: Parent document.
            section: Parent logical section.
            heading_index: Optional heading block to prepend to the first subchunk.
            block_index: Oversized block index in the document.
            block: Oversized structured block.
            block_spans: Character spans for document blocks.
            max_chunk_chars: Maximum allowed chunk size.
            chunk_index_offset: Chunk index offset for generated subchunks.

        Returns:
            Split chunks for oversized code/data blocks, or an empty list when
            the block should be handled by regular section chunking.
        """
        if block.block_type != BlockType.CODE_BLOCK:
            return []

        block_start, block_end = block_spans[block_index]
        rendered_block_text = doc.text[block_start:block_end]
        heading_start = block_spans[heading_index][0] if heading_index is not None else None
        heading_overhead = 0
        if heading_start is not None:
            heading_overhead = block_start - heading_start

        target_block_chars = max(max_chunk_chars - heading_overhead, 200)
        line_groups = self._split_rendered_lines(rendered_block_text, target_block_chars)
        if len(line_groups) <= 1:
            return []

        chunks: list[Chunk] = []
        content_offset = 0
        for group_index, (group_text, consumed_length) in enumerate(line_groups):
            start_char = block_start + content_offset
            end_char = start_char + len(group_text)
            if group_index == 0 and heading_start is not None:
                start_char = heading_start
                group_text = doc.text[start_char:end_char]
            content_offset += consumed_length

            metadata = ChunkMetadata(
                crate=doc.metadata.crate,
                item_path=doc.metadata.item_path,
                item_type=doc.metadata.item_type,
                rust_version=doc.metadata.rust_version,
                url=doc.metadata.url,
                section=section.section_path[-1] if section.section_path else None,
                section_path=section.section_path,
                anchor=section.anchor,
                chunk_index=chunk_index_offset + group_index,
                start_char=start_char,
                end_char=end_char,
                doc_title=doc.title,
                doc_source_path=doc.source_path,
            )
            chunks.append(
                Chunk(
                    chunk_id=Chunk.generate_id(doc.doc_id, chunk_index_offset + group_index),
                    doc_id=doc.doc_id,
                    text=group_text,
                    metadata=metadata,
                )
            )

        return chunks

    def _split_rendered_lines(
        self,
        rendered_text: str,
        target_chars: int,
    ) -> list[tuple[str, int]]:
        """
        Split a large rendered block by line groups.

        Args:
            rendered_text: Exact rendered block text from the source document.
            target_chars: Target maximum size for one split chunk.

        Returns:
            List of `(rendered_text, consumed_length)` tuples.
        """
        lines = rendered_text.splitlines(keepends=True)
        if not lines:
            return [(rendered_text, len(rendered_text))]

        groups: list[tuple[str, int]] = []
        current = ""
        for line in lines:
            if current and len(current) + len(line) > target_chars:
                groups.append((current, len(current)))
                current = line
                continue
            current += line

        if current:
            groups.append((current, len(current)))

        return groups

    def _fallback_chunk(self, doc: Document) -> Chunk:
        """
        Create a single fallback chunk when structured blocks are unavailable.

        Args:
            doc: Source document.

        Returns:
            One chunk covering the whole document text.
        """
        metadata = ChunkMetadata(
            crate=doc.metadata.crate,
            item_path=doc.metadata.item_path,
            item_type=doc.metadata.item_type,
            rust_version=doc.metadata.rust_version,
            url=doc.metadata.url,
            section=doc.title,
            section_path=[doc.title],
            anchor=None,
            chunk_index=0,
            start_char=0,
            end_char=len(doc.text),
            doc_title=doc.title,
            doc_source_path=doc.source_path,
        )
        return Chunk(
            chunk_id=Chunk.generate_id(doc.doc_id, 0),
            doc_id=doc.doc_id,
            text=doc.text,
            metadata=metadata,
        )

    def _merge_small_chunks(
        self,
        doc: Document,
        chunks: list[Chunk],
        max_chunk_chars: int,
    ) -> list[Chunk]:
        """
        Conservatively merge undersized adjacent chunks when they share context.

        Args:
            doc: Parent document.
            chunks: Chunks produced from section splitting.
            max_chunk_chars: Maximum allowed merged chunk size.

        Returns:
            Chunk list with obvious tiny parent-intro chunks merged forward.
        """
        if len(chunks) < 2:
            return chunks

        merged: list[Chunk] = []
        current = chunks[0]
        for next_chunk in chunks[1:]:
            if self._should_merge_chunks(current, next_chunk, max_chunk_chars):
                current = self._merge_two_chunks(doc, current, next_chunk)
                continue
            merged.append(current)
            current = next_chunk

        merged.append(current)
        return [chunk for chunk in merged if not self._should_drop_chunk(chunk)]

    def _should_merge_chunks(
        self,
        left: Chunk,
        right: Chunk,
        max_chunk_chars: int,
    ) -> bool:
        """
        Decide whether two adjacent chunks should be merged.

        Args:
            left: Earlier chunk.
            right: Later adjacent chunk.
            max_chunk_chars: Maximum allowed merged chunk size.

        Returns:
            `True` for safe, context-preserving merges.
        """
        if len(left.text) >= self.min_chunk_chars:
            return False

        merged_length = len(left.text) + 2 + len(right.text)
        if merged_length > max_chunk_chars:
            return False

        left_path = left.metadata.section_path or []
        right_path = right.metadata.section_path or []
        if not left_path or not right_path:
            return False

        if left_path == right_path or self._is_prefix_path(left_path, right_path):
            return True

        crate = left.metadata.crate
        if crate in {Crate.BOOK, Crate.CARGO, Crate.REFERENCE}:
            if (
                len(left.text) < self.min_chunk_chars
                and len(right.text) < self.min_chunk_chars
                and self._share_parent_path(left_path, right_path)
            ):
                return True

        return False

    def _merge_two_chunks(
        self,
        doc: Document,
        left: Chunk,
        right: Chunk,
    ) -> Chunk:
        """
        Merge two adjacent chunks into one combined chunk.

        Args:
            doc: Parent document.
            left: Earlier chunk.
            right: Later adjacent chunk.

        Returns:
            Combined chunk with merged text and widened source span.
        """
        left_path = left.metadata.section_path or []
        right_path = right.metadata.section_path or []
        if self._is_prefix_path(left_path, right_path):
            section = right.metadata.section
            section_path = right.metadata.section_path
            anchor = right.metadata.anchor or left.metadata.anchor
        elif self._share_parent_path(left_path, right_path):
            section_path = left_path[:-1]
            section = section_path[-1] if section_path else left.metadata.section
            anchor = left.metadata.anchor or right.metadata.anchor
        else:
            section = left.metadata.section
            section_path = left.metadata.section_path
            anchor = left.metadata.anchor or right.metadata.anchor

        metadata = left.metadata.model_copy(
            update={
                "section": section,
                "section_path": section_path,
                "anchor": anchor,
                "start_char": min(left.metadata.start_char, right.metadata.start_char),
                "end_char": max(left.metadata.end_char, right.metadata.end_char),
            }
        )
        merged_start = metadata.start_char
        merged_end = metadata.end_char
        return Chunk(
            chunk_id=left.chunk_id,
            doc_id=doc.doc_id,
            text=doc.text[merged_start:merged_end],
            metadata=metadata,
        )

    def _reindex_chunks(self, doc: Document, chunks: list[Chunk]) -> list[Chunk]:
        """
        Rebuild chunk ids and chunk indexes after merges.

        Args:
            doc: Parent document.
            chunks: Chunk list in final document order.

        Returns:
            Reindexed chunk list.
        """
        reindexed: list[Chunk] = []
        for chunk_index, chunk in enumerate(chunks):
            metadata = chunk.metadata.model_copy(update={"chunk_index": chunk_index})
            reindexed.append(
                Chunk(
                    chunk_id=Chunk.generate_id(doc.doc_id, chunk_index),
                    doc_id=doc.doc_id,
                    text=chunk.text,
                    metadata=metadata,
                )
            )
        return reindexed

    def _is_prefix_path(self, prefix: list[str], candidate: list[str]) -> bool:
        """
        Check whether one section path is a strict prefix of another.

        Args:
            prefix: Potential parent section path.
            candidate: Potential child section path.

        Returns:
            `True` when `prefix` is a strict prefix of `candidate`.
        """
        if len(prefix) >= len(candidate):
            return False
        return candidate[: len(prefix)] == prefix

    def _share_parent_path(self, left: list[str], right: list[str]) -> bool:
        """
        Check whether two section paths are sibling sections under one parent.

        Args:
            left: Left chunk section path.
            right: Right chunk section path.

        Returns:
            `True` when both paths share the same parent and differ at the leaf.
        """
        if len(left) != len(right):
            return False
        if len(left) < 3:
            return False
        return left[:-1] == right[:-1] and left[-1] != right[-1]

    def _should_drop_chunk(self, chunk: Chunk) -> bool:
        """
        Drop known low-value chunks that carry almost no retrieval value.

        Args:
            chunk: Candidate chunk after merge.

        Returns:
            `True` when the chunk should be excluded from output.
        """
        section_path = chunk.metadata.section_path or []
        if (
            chunk.metadata.crate == Crate.CARGO
            and chunk.metadata.doc_source_path.replace("\\", "/") == "cargo/CHANGELOG.html"
            and len(section_path) == 2
            and len(chunk.text) < self.min_chunk_chars
        ):
            lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
            if len(lines) <= 2:
                return True
        return False


def chunk_documents(
    docs: list[Document],
    output_file: Optional[Union[Path, str]] = None,
    max_chunk_chars: int = 1400,
) -> list[Chunk]:
    """
    Chunk cleaned documents and optionally persist JSONL output.

    Args:
        docs: Cleaned and deduplicated documents.
        output_file: Optional JSONL output path for chunks.
        max_chunk_chars: Default maximum chunk size in characters.

    Returns:
        Ordered list of generated chunks.
    """
    logger.info("Chunking %s documents...", len(docs))
    chunker = DocumentChunker(max_chunk_chars=max_chunk_chars)
    chunks: list[Chunk] = []
    for index, doc in enumerate(docs, start=1):
        if index % 100 == 0:
            logger.info("Chunked %s/%s documents", index, len(docs))
        chunks.extend(chunker.chunk_document(doc))

    logger.info("Chunk stage complete: produced %s chunks", len(chunks))

    if output_file is not None:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
        logger.info("Saved chunks to %s", out)

    return chunks
