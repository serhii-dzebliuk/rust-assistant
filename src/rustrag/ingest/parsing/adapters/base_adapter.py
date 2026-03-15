from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from ..core import COMMON_REMOVE_SELECTORS, extract_structured_text, remove_noise


class HtmlAdapter(ABC):
    main_selectors: tuple[str, ...] = ("main", "body")
    extra_remove_selectors: tuple[str, ...] = ()

    def select_main(self, soup: BeautifulSoup) -> Tag | None:
        for selector in self.main_selectors:
            node = soup.select_one(selector)
            if isinstance(node, Tag):
                return node
        body = soup.body
        return body if isinstance(body, Tag) else None

    def clean_main(self, root: Tag) -> Tag:
        remove_noise(root, COMMON_REMOVE_SELECTORS + self.extra_remove_selectors)
        return root

    @abstractmethod
    def extract_title(self, soup: BeautifulSoup, file_path: Path) -> str:
        raise NotImplementedError

    def extract_text(self, root: Tag) -> str:
        return extract_structured_text(root)
