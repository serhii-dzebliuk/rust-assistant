"""
Discover HTML documentation files from data/raw/.
Stage 1.2 of the ingest pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..models import Crate

logger = logging.getLogger(__name__)


class DocumentDiscoverer:
    """Discovers HTML documentation files in data/raw/."""

    # Files and patterns to exclude
    EXCLUDE_PATTERNS = {
        # JavaScript files
        "*.js",
        # CSS and style files
        "*.css",
        # Image files
        "*.png", "*.jpg", "*.jpeg", "*.svg", "*.gif", "*.ico",
        # Font files
        "*.woff", "*.woff2", "*.ttf", "*.eot",
        # Other assets
        "*.json", "*.txt", "*.md", ".nojekyll",
        # Special pages
        "search-index*.js",
        "searchindex*.js",
        "sidebar-items*.js",
        "print.html",  # Print version (duplicate content)
        "toc.html",  # Table of contents (navigation)
    }

    # Directories to exclude
    EXCLUDE_DIRS = {
        "theme", "css", "fonts", "FontAwesome", "images", "img",
        ".git", ".venv", "__pycache__",
        "first-edition", "second-edition", "2018-edition",  # Old book editions
    }

    def __init__(self, raw_data_dir: Path):
        """
        Initialize discoverer.

        Args:
            raw_data_dir: Path to data/raw/ directory
        """
        self.raw_data_dir = Path(raw_data_dir).resolve()  # Convert to absolute path
        if not self.raw_data_dir.exists():
            raise ValueError(f"Raw data directory does not exist: {raw_data_dir}")

    def discover(
        self,
        crates: Optional[list[Crate]] = None,
        limit: Optional[int] = None,
    ) -> list[Path]:
        """
        Discover HTML files in raw data directory.

        Args:
            crates: List of crates to include (None = all)
            limit: Maximum number of files to return (None = all)

        Returns:
            List of paths to HTML files, relative to raw_data_dir
        """
        if crates is None:
            crates = [Crate.STD, Crate.BOOK, Crate.CARGO, Crate.REFERENCE]

        html_files = []
        for crate in crates:
            crate_dir = self.raw_data_dir / crate.value

            if not crate_dir.exists():
                logger.warning(f"Crate directory not found: {crate_dir}")
                continue

            crate_files = self._discover_in_directory(crate_dir, crate)
            html_files.extend(crate_files)

            if limit and len(html_files) >= limit:
                html_files = html_files[:limit]
                break

        logger.info(f"Discovered {len(html_files)} HTML files")
        return html_files

    def _discover_in_directory(self, directory: Path, crate: Crate) -> list[Path]:
        """
        Recursively discover HTML files in a directory.

        Args:
            directory: Directory to search
            crate: Crate this directory belongs to

        Returns:
            List of HTML file paths
        """
        html_files = []

        for path in directory.rglob("*.html"):
            # Skip excluded directories
            if any(excluded_dir in path.parts for excluded_dir in self.EXCLUDE_DIRS):
                continue

            # Skip excluded patterns
            if any(path.match(pattern) for pattern in self.EXCLUDE_PATTERNS):
                continue

            # Skip if file is not readable
            if not path.is_file():
                continue

            html_files.append(path)

        logger.debug(f"Found {len(html_files)} files in {crate.value}/")
        return html_files

    def get_relative_path(self, absolute_path: Path) -> Path:
        """
        Get path relative to raw_data_dir.

        Args:
            absolute_path: Absolute file path

        Returns:
            Path relative to raw_data_dir
        """
        return absolute_path.relative_to(self.raw_data_dir)

    def get_crate_from_path(self, path: Path) -> Crate:
        """
        Determine crate from file path.

        Args:
            path: File path (absolute or relative to raw_data_dir)

        Returns:
            Crate enum value
        """
        # Get relative path if absolute
        if path.is_absolute():
            try:
                path = self.get_relative_path(path)
            except ValueError:
                logger.debug(f"Could not get relative path for {path}")
                return Crate.UNKNOWN

        # First part of path should be crate name
        parts = path.parts
        if not parts:
            return Crate.UNKNOWN

        crate_name = parts[0]
        try:
            return Crate(crate_name)
        except ValueError:
            logger.debug(f"Unknown crate name: {crate_name}")
            return Crate.UNKNOWN


def discover_documents(
    raw_data_dir: Path | str = "data/raw",
    crates: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[Path]:
    """
    Convenient function to discover HTML documents.

    Args:
        raw_data_dir: Path to raw data directory
        crates: List of crate names to include (e.g., ["std", "book"])
        limit: Maximum number of files to return

    Returns:
        List of HTML file paths

    Example:
        >>> files = discover_documents(crates=["std"], limit=100)
        >>> print(f"Found {len(files)} files")
    """
    raw_data_dir = Path(raw_data_dir)

    # Convert crate names to Crate enums
    crate_enums = None
    if crates:
        crate_enums = []
        for crate_name in crates:
            try:
                crate_enums.append(Crate(crate_name))
            except ValueError:
                logger.warning(f"Unknown crate: {crate_name}")

    discoverer = DocumentDiscoverer(raw_data_dir)
    return discoverer.discover(crates=crate_enums, limit=limit)


def main():
    """CLI entry point for discovery."""
    import argparse

    parser = argparse.ArgumentParser(description="Discover Rust documentation HTML files")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default="data/raw",
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--crate",
        type=str,
        action="append",
        choices=["std", "book", "cargo", "reference"],
        help="Crate(s) to include (can be specified multiple times, default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of files to discover",
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
    files = discover_documents(
        raw_data_dir=args.raw_dir,
        crates=args.crate,
        limit=args.limit,
    )

    # Print results
    discoverer = DocumentDiscoverer(args.raw_dir)
    print(f"\nDiscovered {len(files)} HTML files:")
    print("-" * 60)

    # Group by crate
    by_crate = {}
    for file_path in files:
        crate = discoverer.get_crate_from_path(file_path)
        if crate not in by_crate:
            by_crate[crate] = []
        by_crate[crate].append(file_path)

    for crate, paths in sorted(by_crate.items()):
        print(f"\n{crate.value}: {len(paths)} files")
        if args.verbose:
            for path in sorted(paths)[:10]:  # Show first 10
                rel_path = discoverer.get_relative_path(path)
                print(f"  - {rel_path}")
            if len(paths) > 10:
                print(f"  ... and {len(paths) - 10} more")


if __name__ == "__main__":
    main()
