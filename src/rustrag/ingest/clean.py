"""
Clean and normalize parsed documents.
Stage 1.4 of the ingest pipeline.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..models import Crate, Document

logger = logging.getLogger(__name__)


class DocumentCleaner:
    """Lightweight post-parse cleanup.

    Stage 1.3 should already return structured text. Stage 1.4 applies
    deterministic normalization rules that are easy to test and tune.
    """

    MIN_TEXT_LENGTH = 30

    def clean_text(self, text: str, crate: Crate) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.replace("§ ", "")

        if crate == Crate.REFERENCE:
            # Remove leftover rule anchors like [destructors.scope.intro]
            text = re.sub(r"(?m)^\[[a-z0-9_.-]+\]\s*$\n?", "", text)

        # rustdoc <wbr> often creates token splits that should stay contiguous.
        text = re.sub(r"\b([A-Za-z0-9]+)_\s+([A-Za-z0-9]+)\b", r"\1_\2", text)
        text = re.sub(r"`([^`\n]+)`\s+s\b", r"`\1`s", text)

        return text.strip()

    def clean_document(self, doc: Document) -> Document | None:
        cleaned_text = self.clean_text(doc.text, doc.metadata.crate)
        if len(cleaned_text) < self.MIN_TEXT_LENGTH:
            return None
        return doc.model_copy(update={"text": cleaned_text})


def clean_documents(
    docs: list[Document],
    output_file: Path | str | None = None,
) -> list[Document]:
    logger.info("Cleaning %s documents...", len(docs))
    cleaner = DocumentCleaner()
    cleaned: list[Document] = []
    dropped_short = 0
    for doc in docs:
        cleaned_doc = cleaner.clean_document(doc)
        if cleaned_doc is None:
            dropped_short += 1
            continue
        cleaned.append(cleaned_doc)

    logger.info(
        "Clean stage complete: kept %s docs, dropped %s short docs",
        len(cleaned),
        dropped_short,
    )

    if output_file is not None:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for doc in cleaned:
                f.write(doc.to_jsonl() + "\n")
        logger.info("Saved cleaned documents to %s", out)

    return cleaned
