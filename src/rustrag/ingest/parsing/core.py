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
    for selector in selectors:
        for node in root.select(selector):
            node.decompose()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text.strip()


def _text_without_nested_lists(li: Tag) -> str:
    copy = BeautifulSoup(str(li), "lxml")
    for nested in copy.select("ul,ol"):
        nested.decompose()
    root = copy.find("li") or copy
    text = _extract_inline_text(root)
    return re.sub(r"\s+", " ", text).strip()


def _extract_inline_text(node: Tag) -> str:
    copy = BeautifulSoup(str(node), "lxml")

    for code in copy.select("code"):
        text = re.sub(r"\s+", " ", code.get_text(" ", strip=True)).strip()
        code.replace_with(f"`{text}`" if text else "")

    text = copy.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _detect_code_language(pre: Tag) -> str:
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
    # Using a newline separator here breaks rustdoc signatures because many
    # inline spans sit inside <pre>. The default separator preserves the
    # meaningful line structure without inserting artificial line breaks.
    return pre.get_text(strip=False).strip("\n")


def extract_structured_text(root: Tag) -> str:
    blocks: list[str] = []
    tags = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "dt", "dd")
    for el in root.find_all(tags):
        inside_pre = any(parent.name == "pre" for parent in el.parents if isinstance(parent, Tag))
        if inside_pre and el.name != "pre":
            continue
        if el.name == "pre" and inside_pre:
            continue
        if el.name == "p" and isinstance(el.parent, Tag) and el.parent.name == "li":
            continue
        if el.name == "dd" and isinstance(el.parent, Tag) and el.parent.name == "dl":
            text = _extract_inline_text(el)
            if text:
                blocks.append(f"  - {text}")
            continue
        if el.name == "dt" and isinstance(el.parent, Tag) and el.parent.name == "dl":
            text = _extract_inline_text(el)
            if text:
                blocks.append(f"- {text}")
            continue

        if el.name.startswith("h"):
            text = _extract_inline_text(el)
            if text:
                blocks.append(text)
            continue

        if el.name == "p":
            text = _extract_inline_text(el)
            if text and text != "Show Railroad":
                blocks.append(text)
            continue

        if el.name == "li":
            text = _text_without_nested_lists(el)
            if not text or text == "Show Railroad":
                continue
            depth = sum(1 for p in el.parents if isinstance(p, Tag) and p.name in {"ul", "ol"}) - 1
            prefix = "  " * max(depth, 0) + "- "
            blocks.append(f"{prefix}{text}")
            continue

        if el.name == "pre":
            code = _extract_code_block_text(el)
            if code:
                language = _detect_code_language(el)
                blocks.append(f"```{language}\n{code}\n```")

    return normalize_text("\n\n".join(blocks))
