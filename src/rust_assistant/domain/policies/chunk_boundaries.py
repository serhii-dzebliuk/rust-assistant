"""Boundary helpers for safe chunk splitting."""

from __future__ import annotations

import re
from typing import Optional

from rust_assistant.domain.errors import ChunkingError


_MIN_TINY_TAIL_CHARS = 80


def split_rendered_lines(rendered_text: str, target_chars: int) -> list[tuple[str, int]]:
    """Split rendered code text on line boundaries."""

    lines = rendered_text.splitlines(keepends=True)
    if not lines:
        return [(rendered_text, len(rendered_text))]

    groups: list[tuple[str, int]] = []
    current = ""
    for line in lines:
        if len(line) > target_chars:
            raise ChunkingError(
                "Cannot safely split a code block line that exceeds "
                f"{target_chars} characters"
            )
        if current and len(current) + len(line) > target_chars:
            groups.append((current, len(current)))
            current = line
            continue
        current += line

    if current:
        groups.append((current, len(current)))

    return _merge_tiny_tail_groups(groups, target_chars)


def split_text_by_boundaries(rendered_text: str, target_chars: int) -> list[tuple[str, int]]:
    """Split rendered text on semantic boundaries without leaving tiny tails."""

    if len(rendered_text) <= target_chars:
        return [(rendered_text, len(rendered_text))]

    groups: list[tuple[str, int]] = []
    start = 0
    while start < len(rendered_text):
        remaining = len(rendered_text) - start
        if remaining <= target_chars:
            groups.append((rendered_text[start:], remaining))
            break

        split_position = _find_safe_split_position(rendered_text, start, target_chars)
        consumed_length = split_position - start
        groups.append((rendered_text[start:split_position], consumed_length))
        start = split_position

    return _merge_tiny_tail_groups(groups, target_chars)


def is_single_fenced_code_block(text: str) -> bool:
    """Return whether the text is exactly one fenced code block."""

    stripped = text.strip()
    return (
        stripped.startswith("```") and stripped.endswith("```") and stripped.count("```") == 2
    )


def _find_safe_split_position(text: str, start: int, target_chars: int) -> int:
    upper = min(len(text), start + target_chars)
    minimum = start + max(1, min(upper - start, int(target_chars * 0.45)))
    segment = text[start:upper]
    boundary_groups = [
        [start + match.end() for match in re.finditer(r"\n\s*\n", segment)],
        [start + match.end() for match in re.finditer(r"\n", segment)],
        [start + match.end() for match in re.finditer(r"(?<=[.!?])\s+", segment)],
        [start + match.end() for match in re.finditer(r"\s+", segment)],
    ]

    fallback: Optional[int] = None
    for boundaries in boundary_groups:
        usable = [
            boundary
            for boundary in boundaries
            if boundary >= minimum and not _inside_fenced_code(text, boundary)
        ]
        if usable:
            tail_aware = [
                boundary
                for boundary in usable
                if not _would_leave_tiny_tail(text, boundary, target_chars)
            ]
            if tail_aware:
                return max(tail_aware)
            if fallback is None:
                fallback = min(usable)

        any_usable = [
            boundary for boundary in boundaries if not _inside_fenced_code(text, boundary)
        ]
        if any_usable and fallback is None:
            fallback = max(any_usable)

    if fallback is not None:
        return fallback
    if _inside_fenced_code(text, upper):
        raise ChunkingError("Cannot safely split inside a fenced code block")
    raise ChunkingError("Cannot safely split text span without a semantic boundary")


def _would_leave_tiny_tail(text: str, split_position: int, target_chars: int) -> bool:
    tail_length = len(text[split_position:].strip())
    return 0 < tail_length < _MIN_TINY_TAIL_CHARS and tail_length <= target_chars


def _merge_tiny_tail_groups(
    groups: list[tuple[str, int]],
    target_chars: int,
) -> list[tuple[str, int]]:
    if len(groups) < 2:
        return groups

    tail_text, tail_consumed = groups[-1]
    if len(tail_text.strip()) >= _MIN_TINY_TAIL_CHARS:
        return groups

    previous_text, previous_consumed = groups[-2]
    if len(previous_text) + len(tail_text) > target_chars:
        return groups

    return [
        *groups[:-2],
        (previous_text + tail_text, previous_consumed + tail_consumed),
    ]


def _inside_fenced_code(text: str, offset: int) -> bool:
    return text[:offset].count("```") % 2 == 1
