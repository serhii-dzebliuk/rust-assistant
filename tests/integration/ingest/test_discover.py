from pathlib import Path

import pytest

from rust_assistant.ingest.discover import DocumentDiscoverer
from rust_assistant.schemas.enums import Crate

pytestmark = pytest.mark.integration


def test_discover_skips_rustdoc_redirect_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    std_dir = raw_dir / "std"
    std_dir.mkdir(parents=True)

    (std_dir / "macro.assert!.html").write_text(
        """<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0;URL=macro.assert.html">
    <title>Redirection</title>
</head>
</html>
""",
        encoding="utf-8",
    )
    real_doc = std_dir / "macro.assert.html"
    real_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>assert in std - Rust</title></head>
<body><main><h1>Macro <span class="macro">assert</span></h1></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.STD])

    assert real_doc in files
    assert std_dir / "macro.assert!.html" not in files


def test_discover_keeps_regular_rustdoc_html_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    std_dir = raw_dir / "std" / "alloc"
    std_dir.mkdir(parents=True)

    real_doc = std_dir / "struct.Layout.html"
    real_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Layout in std::alloc - Rust</title></head>
<body><main><h1>Struct <span class="struct">Layout</span></h1></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.STD])

    assert files == [real_doc]


def test_discover_skips_std_all_items_index(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    std_dir = raw_dir / "std"
    std_dir.mkdir(parents=True)

    skipped_doc = std_dir / "all.html"
    skipped_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>List of all items in this crate</title></head>
<body><main><h1>List of all items</h1><ul><li>std::vec::Vec</li></ul></main></body>
</html>
""",
        encoding="utf-8",
    )
    kept_doc = std_dir / "index.html"
    kept_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>std - Rust</title></head>
<body><main><h1>Crate <span>std</span></h1><p>Standard library docs.</p></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.STD])

    assert kept_doc in files
    assert skipped_doc not in files


def test_discover_skips_book_redirect_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    book_dir = raw_dir / "book"
    book_dir.mkdir(parents=True)

    redirect_page = book_dir / "redirect.html"
    redirect_page.write_text(
        """<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; URL=chapter-1.html">
    <title>Redirecting...</title>
    <link rel="canonical" href="chapter-1.html">
</head>
<body>
    <p>Redirecting to... <a href="chapter-1.html">chapter-1.html</a>.</p>
    <script>window.location.replace("chapter-1.html");</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    real_doc = book_dir / "chapter-1.html"
    real_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Chapter 1 - The Rust Programming Language</title></head>
<body><main><h1>Chapter 1</h1><p>Real content.</p></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.BOOK])

    assert real_doc in files
    assert redirect_page not in files


def test_discover_skips_book_legacy_alias_pages(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    book_dir = raw_dir / "book"
    book_dir.mkdir(parents=True)

    legacy_page = book_dir / "associated-types.html"
    legacy_page.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Associated Types</title></head>
<body>
    <h1 class="title">Associated Types</h1>
    <p><small>There is a new edition of the book and this is an old link.</small></p>
    <blockquote><p>Old alias content.</p></blockquote>
    <p>You can find the latest version of this information <a href="ch20-02-advanced-traits.html">here</a>.</p>
</body>
</html>
""",
        encoding="utf-8",
    )
    real_doc = book_dir / "ch20-02-advanced-traits.html"
    real_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Advanced Traits - The Rust Programming Language</title></head>
<body><main><h1>Advanced Traits</h1><p>Current edition content.</p></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.BOOK])

    assert real_doc in files
    assert legacy_page not in files


def test_discover_keeps_regular_book_html_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    book_dir = raw_dir / "book"
    book_dir.mkdir(parents=True)

    real_doc = book_dir / "chapter-1.html"
    real_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Chapter 1 - The Rust Programming Language</title></head>
<body><main><h1>Chapter 1</h1><p>Real content.</p></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.BOOK])

    assert files == [real_doc]


def test_discover_skips_book_service_files_but_keeps_index(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    book_dir = raw_dir / "book"
    book_dir.mkdir(parents=True)

    skipped_names = ("README.html", "SUMMARY.html", "title-page.html")
    for name in skipped_names:
        (book_dir / name).write_text(
            """<!DOCTYPE html>
<html>
<head><title>The Rust Programming Language</title></head>
<body><main><h1>Service page</h1><p>Should be skipped.</p></main></body>
</html>
""",
            encoding="utf-8",
        )

    kept_index = book_dir / "index.html"
    kept_index.write_text(
        """<!DOCTYPE html>
<html>
<head><title>The Rust Programming Language - The Rust Programming Language</title></head>
<body><main><h1>The Rust Programming Language</h1><p>Edition 2024 and release date info.</p></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.BOOK])

    assert kept_index in files
    for name in skipped_names:
        assert book_dir / name not in files


def test_discover_skips_second_edition_book_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    book_root = raw_dir / "book"
    book_root.mkdir(parents=True)
    excluded_dir = book_root / "second-edition"
    excluded_dir.mkdir(parents=True)

    kept_doc = book_root / "chapter-new.html"
    kept_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>New Chapter</title></head>
<body><main><h1>New chapter</h1><p>Should be kept.</p></main></body>
</html>
""",
        encoding="utf-8",
    )
    skipped_doc = excluded_dir / "chapter-old.html"
    skipped_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Old Edition Chapter</title></head>
<body><main><h1>Old edition</h1><p>Should be skipped.</p></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.BOOK])

    assert kept_doc in files
    assert skipped_doc not in files


def test_discover_skips_reference_redirect_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    ref_dir = raw_dir / "reference"
    ref_dir.mkdir(parents=True)

    redirect_page = ref_dir / "types-redirect.html"
    redirect_page.write_text(
        """<script>
(function() {
    var fragments = {"#foo": "types/numeric.html"};
    var target = fragments[window.location.hash];
    if (target) {
        window.location.replace(target);
    }
})();
</script>
""",
        encoding="utf-8",
    )
    real_doc = ref_dir / "types.html"
    real_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Types - The Rust Reference</title></head>
<body><main><h1>Types</h1><p>Real content.</p></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.REFERENCE])

    assert real_doc in files
    assert redirect_page not in files


def test_discover_skips_html_pages_without_meaningful_main_content(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    ref_dir = raw_dir / "reference"
    ref_dir.mkdir(parents=True)

    for name, heading in (
        ("appendices.html", "Appendices"),
        ("lexical-structure.html", "Lexical structure"),
        ("type-system.html", "Type system"),
    ):
        (ref_dir / name).write_text(
            f"""<!DOCTYPE html>
<html>
<head><title>{heading} - The Rust Reference</title></head>
<body><main><h1>{heading}</h1></main></body>
</html>
""",
            encoding="utf-8",
        )

    kept_doc = ref_dir / "destructors.html"
    kept_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Destructors - The Rust Reference</title></head>
<body><main><h1>Destructors</h1><p>Real content.</p></main></body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.REFERENCE])

    assert kept_doc in files
    assert ref_dir / "appendices.html" not in files
    assert ref_dir / "lexical-structure.html" not in files
    assert ref_dir / "type-system.html" not in files


def test_discover_keeps_placeholder_pages_with_real_text(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    ref_dir = raw_dir / "reference" / "names"
    ref_dir.mkdir(parents=True)

    placeholder_doc = ref_dir / "name-resolution.html"
    placeholder_doc.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Name resolution - The Rust Reference</title></head>
<body>
<main>
    <h1>Name resolution</h1>
    <p><strong>Note</strong> This is a placeholder for future expansion.</p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )

    discoverer = DocumentDiscoverer(raw_dir)

    files = discoverer.discover(crates=[Crate.REFERENCE])

    assert files == [placeholder_doc]
