"""
Parse HTML documentation files into Document objects.
Stage 1.3 of the ingest pipeline.

Extracts:
- Title
- Main content (without sidebar/nav/footer)
- Item path (from breadcrumbs/headers)
- Metadata (crate, item_type, etc.)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag

from ..models import Crate, Document, DocumentMetadata, ItemType

logger = logging.getLogger(__name__)


class RustDocParser:
    """Parser for Rust HTML documentation."""

    # Selectors for different content areas
    MAIN_CONTENT_SELECTORS = [
        "main.content",  # rustdoc standard
        "div#content",
        "main",
        "div.docblock",
    ]

    # Elements to remove (navigation, sidebars, etc.)
    REMOVE_SELECTORS = [
        "nav",
        "aside",
        "header",
        "footer",
        ".sidebar",
        "#sidebar",
        ".mobile-topbar",
        ".out-of-band",  # rustdoc navigation links
        ".search-form",
        ".theme-picker",
        ".nav-wide-wrapper",
        "script",
        "style",
        "noscript",
    ]

    # Item type detection patterns
    ITEM_TYPE_PATTERNS = {
        ItemType.FUNCTION: [r"\bfn\s+\w+", r"Function\s+\w+", r"function\."],
        ItemType.STRUCT: [r"\bstruct\s+\w+", r"Struct\s+\w+", r"struct\."],
        ItemType.TRAIT: [r"\btrait\s+\w+", r"Trait\s+\w+", r"trait\."],
        ItemType.METHOD: [r"\bimpl\s+.*\s+for\s+", r"Method\s+\w+"],
        ItemType.IMPL: [r"\bimpl\s+\w+", r"Implementation"],
        ItemType.MODULE: [r"\bmod\s+\w+", r"Module\s+\w+", r"module\."],
        ItemType.MACRO: [r"\bmacro\s+\w+", r"Macro\s+\w+", r"macro\."],
        ItemType.ENUM: [r"\benum\s+\w+", r"Enum\s+\w+", r"enum\."],
        ItemType.CONSTANT: [r"\bconst\s+\w+", r"Constant\s+\w+", r"constant\."],
        ItemType.TYPE_ALIAS: [r"\btype\s+\w+", r"Type Definition"],
    }

    def __init__(self, raw_data_dir: Path):
        """
        Initialize parser.

        Args:
            raw_data_dir: Path to data/raw/ directory
        """
        self.raw_data_dir = Path(raw_data_dir).resolve()

    def parse_file(self, file_path: Path, crate: Optional[Crate] = None) -> Optional[Document]:
        """
        Parse HTML file into Document.

        Args:
            file_path: Path to HTML file
            crate: Crate this file belongs to (optional, will be detected from path)

        Returns:
            Document object or None if parsing fails
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                html_content = f.read()

            # Parse HTML
            soup = BeautifulSoup(html_content, "lxml")

            # Determine crate from path if not provided (needed for title extraction)
            if crate is None:
                crate = self._detect_crate_from_path(file_path)

            # Extract components
            title = self._extract_title(soup, file_path, crate)
            if not title:
                logger.warning(f"No title found in {file_path}")
                return None

            main_text = self._extract_main_content(soup)
            if not main_text or len(main_text.strip()) < 50:
                logger.warning(f"No meaningful content in {file_path}")
                return None

            # Extract metadata
            item_path = self._extract_item_path(soup, file_path, crate)
            item_type = self._detect_item_type(soup, title, main_text)
            breadcrumbs = self._extract_breadcrumbs(soup)

            # Get relative path
            try:
                source_path = str(file_path.relative_to(self.raw_data_dir))
            except ValueError:
                source_path = str(file_path)

            # Create metadata
            metadata = DocumentMetadata(
                crate=crate,
                item_path=item_path,
                item_type=item_type,
                raw_html_path=str(file_path),
                breadcrumbs=breadcrumbs,
            )

            # Create document
            doc_id = Document.generate_id(source_path, title)
            document = Document(
                doc_id=doc_id,
                title=title,
                source_path=source_path,
                text=main_text,
                metadata=metadata,
            )

            return document

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}", exc_info=True)
            return None

    def _detect_crate_from_path(self, file_path: Path) -> Crate:
        """
        Detect crate from file path.

        Args:
            file_path: HTML file path

        Returns:
            Crate enum
        """
        try:
            rel_path = file_path.relative_to(self.raw_data_dir)
            first_part = rel_path.parts[0] if rel_path.parts else ""

            # Try exact match
            try:
                return Crate(first_part)
            except ValueError:
                pass

            # Try partial matches
            first_part_lower = first_part.lower()
            if "book" in first_part_lower:
                return Crate.BOOK
            elif "reference" in first_part_lower:
                return Crate.REFERENCE
            elif "cargo" in first_part_lower:
                return Crate.CARGO

        except ValueError:
            pass

        return Crate.UNKNOWN

    def _extract_title(self, soup: BeautifulSoup, file_path: Path, crate: Crate) -> Optional[str]:
        """
        Extract page title.

        Strategy:
        - For std: Use first <h1> in <main>
          Example: "Keyword async" from `<h1>Keyword <span>async</span></h1>`
        - For book/cargo/reference: Use <title> tag (preserves context)
          Example: "Variables and Mutability - The Rust Programming Language"

        Args:
            soup: Parsed HTML
            file_path: Path to HTML file
            crate: Detected crate
        """

        # For std-family crates: extract h1 from main content
        if crate in (Crate.STD, Crate.CORE, Crate.ALLOC, Crate.PROC_MACRO, Crate.TEST):
            main_tag = soup.find("main")
            if main_tag:
                h1_tag = main_tag.find("h1")
                if h1_tag:
                    # Clone h1 to avoid modifying original
                    h1_copy = BeautifulSoup(str(h1_tag), "lxml")
                    # Remove button elements (e.g., "Copy item path")
                    for button in h1_copy.find_all("button"):
                        button.decompose()
                    # Get text, stripping nested tags like <span>
                    return h1_copy.get_text(separator=" ", strip=True)
        else:
            # For book/cargo/reference: use title tag (preserves source context)
            title_tag = soup.find("title")
            if title_tag:
                return title_tag.get_text(strip=True)

        # Last resort: filename
        return file_path.stem

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content text, removing navigation/sidebars.

        Returns:
            Cleaned text content
        """
        # Find main content area
        main_content = None
        for selector in self.MAIN_CONTENT_SELECTORS:
            parts = selector.split(".")
            if len(parts) == 2:
                tag_name, class_name = parts
                main_content = soup.find(tag_name, class_=class_name)
            else:
                main_content = soup.find(selector)

            if main_content:
                break

        if not main_content:
            # Fallback to body
            main_content = soup.find("body")

        if not main_content:
            return ""

        # Clone to avoid modifying original
        content_copy = BeautifulSoup(str(main_content), "lxml")

        # Remove unwanted elements
        for selector in self.REMOVE_SELECTORS:
            for element in content_copy.select(selector):
                element.decompose()

        # Extract text
        text = content_copy.get_text(separator="\n", strip=False)

        # Clean up whitespace
        text = self._normalize_whitespace(text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.

        - Remove excessive blank lines (max 2 consecutive)
        - Strip trailing whitespace from lines
        - Preserve code block structure
        """
        lines = text.split("\n")
        cleaned_lines = []
        blank_count = 0

        for line in lines:
            stripped = line.rstrip()

            if not stripped:
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 blank lines
                    cleaned_lines.append("")
            else:
                blank_count = 0
                cleaned_lines.append(stripped)

        # Remove leading/trailing blank lines
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> Optional[list[str]]:
        """
        Extract breadcrumb navigation path.

        Examples:
        - ["std", "vec", "Vec"]
        - ["book", "ch04-01-what-is-ownership"]
        """
        breadcrumbs = []

        # Try rustdoc breadcrumbs
        breadcrumb_nav = soup.find("nav", class_="sub")
        if breadcrumb_nav:
            links = breadcrumb_nav.find_all("a")
            breadcrumbs = [link.get_text(strip=True) for link in links]

        # Try other breadcrumb patterns
        if not breadcrumbs:
            breadcrumb_div = soup.find("div", class_="breadcrumbs")
            if breadcrumb_div:
                links = breadcrumb_div.find_all("a")
                breadcrumbs = [link.get_text(strip=True) for link in links]

        return breadcrumbs if breadcrumbs else None

    def _extract_item_path(
        self, soup: BeautifulSoup, file_path: Path, crate: Crate
    ) -> Optional[str]:
        """
        Extract item path (e.g., "std::vec::Vec::push").

        Uses:
        1. Breadcrumbs
        2. File path structure
        3. Page title/heading
        """
        breadcrumbs = self._extract_breadcrumbs(soup)
        if breadcrumbs:
            # Join breadcrumbs with ::
            return "::".join(breadcrumbs)

        # Try to construct from file path
        try:
            rel_path = file_path.relative_to(self.raw_data_dir)
            parts = list(rel_path.parts[1:])  # Skip crate dir
            parts[-1] = rel_path.stem  # Remove .html extension

            # Clean up common patterns
            parts = [p for p in parts if p not in [".", "..", "index"]]

            if parts:
                return f"{crate.value}::" + "::".join(parts)
        except ValueError:
            pass

        return None

    def _detect_item_type(self, soup: BeautifulSoup, title: str, content: str) -> ItemType:
        """
        Detect type of Rust item (function, struct, trait, etc.).

        Uses:
        1. CSS classes on main elements
        2. Title patterns
        3. Content patterns
        """
        # Check for book sections
        if "book" in title.lower() or "chapter" in title.lower():
            return ItemType.BOOK_SECTION

        # Try CSS classes
        main_content = soup.find("main")
        if main_content:
            classes = main_content.get("class", [])
            for cls in classes:
                if "function" in cls:
                    return ItemType.FUNCTION
                elif "struct" in cls:
                    return ItemType.STRUCT
                elif "trait" in cls:
                    return ItemType.TRAIT
                elif "enum" in cls:
                    return ItemType.ENUM
                elif "macro" in cls:
                    return ItemType.MACRO
                elif "mod" in cls or "module" in cls:
                    return ItemType.MODULE

        # Try title patterns
        title_lower = title.lower()
        for item_type, patterns in self.ITEM_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    return item_type

        # Try content patterns (first 500 chars)
        content_sample = content[:500].lower()
        for item_type, patterns in self.ITEM_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_sample, re.IGNORECASE):
                    return item_type

        return ItemType.UNKNOWN


def parse_documents(
    html_files: list[Path],
    raw_data_dir: Path | str = "data/raw",
    output_file: Optional[Path | str] = None,
) -> list[Document]:
    """
    Parse multiple HTML files into Documents.

    Args:
        html_files: List of HTML file paths
        raw_data_dir: Path to raw data directory
        output_file: Optional path to save JSONL output

    Returns:
        List of parsed Documents

    Example:
        >>> from rustrag.ingest.discover import discover_documents
        >>> files = discover_documents(limit=100)
        >>> docs = parse_documents(files, output_file="data/processed/docs.jsonl")
        >>> print(f"Parsed {len(docs)} documents")
    """
    raw_data_dir = Path(raw_data_dir)
    parser = RustDocParser(raw_data_dir)

    documents = []
    failed = 0

    logger.info(f"Parsing {len(html_files)} HTML files...")

    for i, file_path in enumerate(html_files, 1):
        if i % 100 == 0:
            logger.info(f"Parsed {i}/{len(html_files)} files ({failed} failed)")

        doc = parser.parse_file(file_path)
        if doc:
            documents.append(doc)
        else:
            failed += 1

    logger.info(f"Parsed {len(documents)} documents ({failed} failed)")

    # Save to JSONL if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc.to_jsonl() + "\n")

        logger.info(f"Saved documents to {output_file}")

    return documents


def main():
    """CLI entry point for parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse Rust HTML documentation into Documents")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default="data/raw",
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="data/processed/docs.jsonl",
        help="Output JSONL file (default: data/processed/docs.jsonl)",
    )
    parser.add_argument(
        "--crate",
        type=str,
        action="append",
        choices=["std", "book", "cargo", "reference"],
        help="Crate(s) to parse (can be specified multiple times)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of files to parse",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Discover files
    from .discover import discover_documents

    logger.info("Discovering HTML files...")
    html_files = discover_documents(
        raw_data_dir=args.raw_dir,
        crates=args.crate,
        limit=args.limit,
    )

    if not html_files:
        logger.error("No HTML files found!")
        return 1

    # Parse documents
    documents = parse_documents(
        html_files=html_files,
        raw_data_dir=args.raw_dir,
        output_file=args.output,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Parsing Summary")
    print(f"{'='*60}")
    print(f"Total files: {len(html_files)}")
    print(f"Parsed successfully: {len(documents)}")
    print(f"Failed: {len(html_files) - len(documents)}")
    print(f"Output: {args.output}")

    # Show sample
    if documents and args.verbose:
        print(f"\n{'='*60}")
        print("Sample Document:")
        print(f"{'='*60}")
        doc = documents[0]
        print(f"ID: {doc.doc_id}")
        print(f"Title: {doc.title}")
        print(f"Source: {doc.source_path}")
        print(f"Crate: {doc.metadata.crate}")
        print(f"Item Path: {doc.metadata.item_path}")
        print(f"Item Type: {doc.metadata.item_type}")
        print(f"Text length: {len(doc.text)} chars")
        print(f"Text preview: {doc.text[:200]}...")

    return 0


if __name__ == "__main__":
    exit(main())
