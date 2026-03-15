from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from rustrag.ingest.parsing.adapters.base_adapter import HtmlAdapter

class ReferenceAdapter(HtmlAdapter):
    main_selectors = ("main", "main.content", "div#content")
    extra_remove_selectors = (
        "div.rule",
        ".rule-link",
        "wbr",
        ".toctitle",
        ".menu-title",
    )

    def extract_title(self, soup: BeautifulSoup, file_path: Path) -> str:
        if soup.title:
            return soup.title.get_text(" ", strip=True)
        h1 = soup.select_one("main h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return file_path.stem
