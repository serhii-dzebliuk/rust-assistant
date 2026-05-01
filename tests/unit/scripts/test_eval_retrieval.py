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


def test_parse_args_defaults_to_production_retrieval_shape(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(sys, "argv", ["eval_retrieval.py"])

    args = module._parse_args()

    assert args.retrieval_limit == 20
    assert args.reranking_limit == 10


def test_run_eval_sends_use_reranking_flag(monkeypatch):
    module = _load_module()
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "std::keyword::async",
                        "source_path": "std/keyword.async.html",
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def post(self, url, json):
            calls.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(module.httpx, "Client", FakeClient)
    case = module.EvalCase(
        id="async",
        question="What is async?",
        expected=[{"title_contains": "async"}],
    )

    results = module.run_eval(
        cases=[case],
        base_url="http://api",
        retrieval_limit=50,
        reranking_limit=10,
        use_reranking=False,
        timeout_seconds=1.0,
    )

    assert results[0].rank == 1
    assert calls == [
        {
            "url": "http://api/search",
            "json": {
                "query": "What is async?",
                "retrieval_limit": 50,
                "reranking_limit": 10,
                "use_reranking": False,
            },
        }
    ]


def test_run_eval_omits_use_reranking_for_default_rerank_mode(monkeypatch):
    module = _load_module()
    calls = []

    class FakeResponse:
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def post(self, url, json):
            calls.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(module.httpx, "Client", FakeClient)
    case = module.EvalCase(id="async", question="What is async?", expected=[])

    module.run_eval(
        cases=[case],
        base_url="http://api",
        retrieval_limit=50,
        reranking_limit=10,
        use_reranking=True,
        timeout_seconds=1.0,
    )

    assert calls[0]["json"] == {
        "query": "What is async?",
        "retrieval_limit": 50,
        "reranking_limit": 10,
    }


def test_print_compare_report_marks_improvements_and_regressions(capsys):
    module = _load_module()
    improved = module.EvalCase(id="improved", question="q1", expected=[])
    regressed = module.EvalCase(id="regressed", question="q2", expected=[])
    vector_results = [
        module.CaseResult(
            case=improved,
            rank=3,
            top_score=0.7,
            top_title="old",
            top_source_path="old.html",
            elapsed_ms=10.0,
        ),
        module.CaseResult(
            case=regressed,
            rank=1,
            top_score=0.9,
            top_title="good",
            top_source_path="good.html",
            elapsed_ms=10.0,
        ),
    ]
    rerank_results = [
        module.CaseResult(
            case=improved,
            rank=1,
            top_score=0.95,
            top_title="new",
            top_source_path="new.html",
            elapsed_ms=50.0,
        ),
        module.CaseResult(
            case=regressed,
            rank=None,
            top_score=0.2,
            top_title="bad",
            top_source_path="bad.html",
            elapsed_ms=50.0,
        ),
    ]

    module.print_compare_report(
        vector_results=vector_results,
        rerank_results=rerank_results,
        reranking_limit=10,
    )

    output = capsys.readouterr().out
    assert "delta_mrr" in output
    assert "improvements:" in output
    assert "regressions:" in output
    assert "vector_rank=3 rerank_rank=1" in output
    assert "vector_rank=1 rerank_rank=miss" in output
