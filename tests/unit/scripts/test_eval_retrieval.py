from pathlib import Path
import importlib.util
import sys

import pytest


pytestmark = pytest.mark.unit


def _load_module():
    path = Path("scripts/eval_retrieval.py")
    spec = importlib.util.spec_from_file_location("eval_retrieval", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_matching_rank_matches_exact_and_contains_criteria():
    module = _load_module()
    hits = [
        {
            "title": "std::future",
            "source_path": "std/future/index.html",
            "item_path": "std::future",
        },
        {
            "title": "std::keyword::async",
            "source_path": "std/keyword.async.html",
            "item_path": "std::keyword::async",
        },
    ]

    rank = module.first_matching_rank(
        hits,
        [
            {"source_path": "missing.html"},
            {"title_contains": "keyword::async"},
        ],
    )

    assert rank == 2


def test_first_matching_rank_returns_none_when_no_hit_matches():
    module = _load_module()

    assert module.first_matching_rank([{"title": "std::future"}], [{"title": "Vec"}]) is None
