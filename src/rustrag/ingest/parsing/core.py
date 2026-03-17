"""Core HTML-to-text extraction utilities for parsing adapters."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag


COMMON_REMOVE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "nav",
    "aside",
    "header",
    "footer",
    ".sidebar",
    "#sidebar",
    ".mobile-topbar",
    ".search-form",
    ".theme-picker",
    ".nav-wide-wrapper",
    ".out-of-band",
    "button",
)


def remove_noise(root: Tag, selectors: tuple[str, ...]) -> None:
    """
    Remove noisy HTML nodes from a root element.

    Args:
        root: Parsed HTML root where cleanup should happen.
        selectors: CSS selectors for nodes that should be removed.

    Returns:
        None. The function mutates `root` in place.

    Example:
        >>> remove_noise(main_tag, ("script", ".sidebar"))
    """
    for selector in selectors:
        for node in root.select(selector):
            node.decompose()


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and punctuation spacing in extracted text.

    Args:
        text: Raw text blocks joined from parsed HTML.

    Returns:
        Cleaned text with normalized newlines and spacing.

    Example:
        >>> normalize_text("Hello  \\n\\n\\nworld")
        'Hello\\n\\nworld'
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text.strip()


def _text_without_nested_lists(list_item: Tag) -> str:
    """
    Extract list-item text while ignoring nested list content.

    Args:
        list_item: `<li>` element from the parsed HTML tree.

    Returns:
        Normalized text for the current list item only.
    """
    copy = BeautifulSoup(str(list_item), "lxml")
    for nested in copy.select("ul,ol"):
        nested.decompose()
    root = copy.find("li") or copy
    text = _extract_inline_text(root)
    return re.sub(r"\s+", " ", text).strip()


def _extract_inline_text(node: Tag) -> str:
    """
    Extract inline text and preserve `<code>` fragments as backticks.

    Args:
        node: Element that may contain inline formatting and code spans.

    Returns:
        Whitespace-normalized text with inline code markers preserved.
    """
    copy = BeautifulSoup(str(node), "lxml")

    for code in copy.select("code"):
        text = re.sub(r"\s+", " ", code.get_text(" ", strip=True)).strip()
        code.replace_with(f"`{text}`" if text else "")

    text = copy.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _detect_code_language(pre: Tag) -> str:
    """
    Detect language label for a `<pre>` block.

    Args:
        pre: Preformatted HTML node potentially containing code classes.

    Returns:
        Markdown fence language. Falls back to `text`.
    """
    candidates: list[str] = []
    code = pre.find("code")
    if isinstance(code, Tag):
        candidates.extend(code.get("class", []))
    candidates.extend(pre.get("class", []))

    for candidate in candidates:
        normalized = candidate.lower()
        if normalized == "rust":
            return "rust"
        if candidate.startswith("language-"):
            language = candidate.removeprefix("language-").lower()
            return {
                "shell": "console",
                "bash": "console",
                "sh": "console",
                "zsh": "console",
            }.get(language, language)

    return "text"


def _extract_code_block_text(pre: Tag) -> str:
    """
    Extract code block text without introducing spurious line breaks.

    Args:
        pre: `<pre>` element from HTML.

    Returns:
        Raw code content suitable for markdown fenced blocks.

    Example:
        >>> _extract_code_block_text(pre_tag)
        'fn main() {\\n    println!(\"hi\");\\n}'
    """
    return pre.get_text(strip=False).strip("\n")


def extract_structured_text(root: Tag) -> str:
    """
    Convert cleaned HTML main content into normalized markdown-like text.

    Args:
        root: Main content element selected by a source adapter.

    Returns:
        Structured plain text preserving headings, lists, and code blocks.

    Example:
        >>> text = extract_structured_text(main_tag)
        >>> text.startswith("Chapter 1")
        True
    """
    blocks: list[str] = []
    tags = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "dt", "dd")
    for element in root.find_all(tags):
        inside_pre = any(
            parent.name == "pre" for parent in element.parents if isinstance(parent, Tag)
        )
        if inside_pre and element.name != "pre":
            continue
        if element.name == "pre" and inside_pre:
            continue
        if (
            element.name == "p"
            and isinstance(element.parent, Tag)
            and element.parent.name == "li"
        ):
            continue

        if (
            element.name == "dd"
            and isinstance(element.parent, Tag)
            and element.parent.name == "dl"
        ):
            text = _extract_inline_text(element)
            if text:
                blocks.append(f"  - {text}")
            continue
        if (
            element.name == "dt"
            and isinstance(element.parent, Tag)
            and element.parent.name == "dl"
        ):
            text = _extract_inline_text(element)
            if text:
                blocks.append(f"- {text}")
            continue

        if element.name.startswith("h"):
            text = _extract_inline_text(element)
            if text:
                blocks.append(text)
            continue

        if element.name == "p":
            text = _extract_inline_text(element)
            if text and text != "Show Railroad":
                blocks.append(text)
            continue

        if element.name == "li":
            text = _text_without_nested_lists(element)
            if not text or text == "Show Railroad":
                continue
            depth = (
                sum(
                    1
                    for parent in element.parents
                    if isinstance(parent, Tag) and parent.name in {"ul", "ol"}
                )
                - 1
            )
            prefix = "  " * max(depth, 0) + "- "
            blocks.append(f"{prefix}{text}")
            continue

        if element.name == "pre":
            code = _extract_code_block_text(element)
            if code:
                language = _detect_code_language(element)
                blocks.append(f"```{language}\n{code}\n```")

    return normalize_text("\n\n".join(blocks))
