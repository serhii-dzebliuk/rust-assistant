"""Evaluate retrieval quality against a small JSONL golden set."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DATASET = Path("data/eval/retrieval_questions.jsonl")
EVAL_MODES = ("rerank", "vector", "compare")


@dataclass(frozen=True)
class EvalCase:
    """One retrieval evaluation question and its acceptable matches."""

    id: str
    question: str
    expected: list[dict[str, str]]


@dataclass(frozen=True)
class CaseResult:
    """Metrics and debug data for one evaluated retrieval case."""

    case: EvalCase
    rank: Optional[int]
    top_score: Optional[float]
    top_title: Optional[str]
    top_source_path: Optional[str]
    elapsed_ms: float
    error: Optional[str] = None

    @property
    def matched(self) -> bool:
        return self.rank is not None


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate metrics for one evaluated retrieval mode."""

    total: int
    matched: int
    hit_rate: float
    mrr: float
    average_latency_ms: float


def main() -> int:
    args = _parse_args()
    cases = load_cases(args.dataset)
    if not cases:
        print(f"No eval cases found in {args.dataset}", file=sys.stderr)
        return 2

    if args.mode == "compare":
        vector_results = run_eval(
            cases=cases,
            base_url=args.base_url,
            retrieval_limit=args.retrieval_limit,
            reranking_limit=args.reranking_limit,
            use_reranking=False,
            timeout_seconds=args.timeout,
        )
        rerank_results = run_eval(
            cases=cases,
            base_url=args.base_url,
            retrieval_limit=args.retrieval_limit,
            reranking_limit=args.reranking_limit,
            use_reranking=True,
            timeout_seconds=args.timeout,
        )
        print_compare_report(
            vector_results=vector_results,
            rerank_results=rerank_results,
            reranking_limit=args.reranking_limit,
        )
        return (
            0
            if _success_rate(rerank_results) >= args.min_success_rate
            else 1
        )

    use_reranking = args.mode == "rerank"
    results = run_eval(
        cases=cases,
        base_url=args.base_url,
        retrieval_limit=args.retrieval_limit,
        reranking_limit=args.reranking_limit,
        use_reranking=use_reranking,
        timeout_seconds=args.timeout,
    )
    print_report(results, mode=args.mode, reranking_limit=args.reranking_limit)
    return 0 if _success_rate(results) >= args.min_success_rate else 1


def load_cases(path: Path) -> list[EvalCase]:
    """Load retrieval eval cases from a JSONL file."""
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            cases.append(
                EvalCase(
                    id=str(payload.get("id", f"line-{line_number}")),
                    question=str(payload["question"]),
                    expected=list(payload["expected"]),
                )
            )
    return cases


def run_eval(
    *,
    cases: list[EvalCase],
    base_url: str,
    retrieval_limit: int,
    reranking_limit: int,
    use_reranking: bool,
    timeout_seconds: float,
) -> list[CaseResult]:
    """Run all retrieval cases against a live /search endpoint."""
    results: list[CaseResult] = []
    search_url = f"{base_url.rstrip('/')}/search"
    with httpx.Client(timeout=timeout_seconds) as client:
        for case in cases:
            started = perf_counter()
            try:
                payload: dict[str, Any] = {
                    "query": case.question,
                    "retrieval_limit": retrieval_limit,
                    "reranking_limit": reranking_limit,
                }
                if not use_reranking:
                    payload["use_reranking"] = False
                response = client.post(
                    search_url,
                    json=payload,
                )
                _raise_for_status_with_body(response)
                hits = response.json().get("results", [])
                rank = first_matching_rank(hits, case.expected)
                elapsed_ms = (perf_counter() - started) * 1000
                top_hit = hits[0] if hits else {}
                results.append(
                    CaseResult(
                        case=case,
                        rank=rank,
                        top_score=_optional_float(top_hit.get("score")),
                        top_title=_optional_str(top_hit.get("title")),
                        top_source_path=_optional_str(top_hit.get("source_path")),
                        elapsed_ms=elapsed_ms,
                    )
                )
            except Exception as exc:
                elapsed_ms = (perf_counter() - started) * 1000
                results.append(
                    CaseResult(
                        case=case,
                        rank=None,
                        top_score=None,
                        top_title=None,
                        top_source_path=None,
                        elapsed_ms=elapsed_ms,
                        error=str(exc),
                    )
                )
    return results


def _raise_for_status_with_body(response: httpx.Response) -> None:
    """Raise an HTTP status error that includes the response body."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        message = f"{exc} body={response.text!r}"
        raise httpx.HTTPStatusError(
            message,
            request=exc.request,
            response=exc.response,
        ) from exc


def first_matching_rank(
    hits: list[dict[str, Any]],
    expected: list[dict[str, str]],
) -> Optional[int]:
    """Return the 1-based rank of the first hit matching any expected criterion."""
    for index, hit in enumerate(hits, start=1):
        if any(_matches_expected(hit, criterion) for criterion in expected):
            return index
    return None


def summarize_results(results: list[CaseResult]) -> EvalSummary:
    """Return aggregate retrieval metrics for one result set."""
    total = len(results)
    matched = sum(1 for result in results if result.matched)
    mrr = sum(1 / result.rank for result in results if result.rank is not None) / total
    average_latency = sum(result.elapsed_ms for result in results) / total
    return EvalSummary(
        total=total,
        matched=matched,
        hit_rate=matched / total,
        mrr=mrr,
        average_latency_ms=average_latency,
    )


def print_report(results: list[CaseResult], *, mode: str, reranking_limit: int) -> None:
    """Print aggregate retrieval metrics and weak cases."""
    summary = summarize_results(results)

    print(f"mode: {mode}")
    print(f"cases: {summary.total}")
    print(
        f"hit_rate@{reranking_limit}: "
        f"{summary.hit_rate:.3f} ({summary.matched}/{summary.total})"
    )
    print(f"mrr@{reranking_limit}: {summary.mrr:.3f}")
    print(f"avg_latency_ms: {summary.average_latency_ms:.1f}")

    weak_results = [result for result in results if not result.matched or result.rank != 1]
    if not weak_results:
        print("weak_cases: none")
        return

    print("weak_cases:")
    for result in weak_results:
        rank = result.rank if result.rank is not None else "miss"
        print(
            f"- {result.case.id}: rank={rank} top_score={result.top_score} "
            f"top={result.top_title!r} source={result.top_source_path!r}"
        )
        if result.error is not None:
            print(f"  error={result.error}")


def print_compare_report(
    *,
    vector_results: list[CaseResult],
    rerank_results: list[CaseResult],
    reranking_limit: int,
) -> None:
    """Print side-by-side vector and reranker retrieval metrics."""
    vector_summary = summarize_results(vector_results)
    rerank_summary = summarize_results(rerank_results)

    print(f"cases: {vector_summary.total}")
    _print_compare_row("vector", vector_summary, reranking_limit=reranking_limit)
    _print_compare_row("rerank", rerank_summary, reranking_limit=reranking_limit)
    print(f"delta_hit_rate: {rerank_summary.hit_rate - vector_summary.hit_rate:+.3f}")
    print(f"delta_mrr: {rerank_summary.mrr - vector_summary.mrr:+.3f}")
    print(
        "delta_avg_latency_ms: "
        f"{rerank_summary.average_latency_ms - vector_summary.average_latency_ms:+.1f}"
    )

    print("case_deltas:")
    improvements: list[tuple[CaseResult, CaseResult]] = []
    regressions: list[tuple[CaseResult, CaseResult]] = []
    for vector_result, rerank_result in zip(vector_results, rerank_results):
        vector_rank = _rank_value(vector_result)
        rerank_rank = _rank_value(rerank_result)
        if rerank_rank < vector_rank:
            improvements.append((vector_result, rerank_result))
        elif rerank_rank > vector_rank:
            regressions.append((vector_result, rerank_result))
        print(_format_case_delta(vector_result, rerank_result))

    _print_delta_group("improvements", improvements)
    _print_delta_group("regressions", regressions)


def _print_compare_row(
    label: str,
    summary: EvalSummary,
    *,
    reranking_limit: int,
) -> None:
    print(
        f"{label}: hit_rate@{reranking_limit}={summary.hit_rate:.3f} "
        f"({summary.matched}/{summary.total}) "
        f"mrr@{reranking_limit}={summary.mrr:.3f} "
        f"avg_latency_ms={summary.average_latency_ms:.1f}"
    )


def _print_delta_group(
    label: str,
    pairs: list[tuple[CaseResult, CaseResult]],
) -> None:
    if not pairs:
        print(f"{label}: none")
        return

    print(f"{label}:")
    for vector_result, rerank_result in pairs:
        print(_format_case_delta(vector_result, rerank_result))


def _format_case_delta(vector_result: CaseResult, rerank_result: CaseResult) -> str:
    return (
        f"- {vector_result.case.id}: "
        f"vector_rank={_rank_label(vector_result)} "
        f"rerank_rank={_rank_label(rerank_result)} "
        f"vector_top={vector_result.top_title!r} "
        f"rerank_top={rerank_result.top_title!r}"
    )


def _rank_value(result: CaseResult) -> int:
    return result.rank if result.rank is not None else sys.maxsize


def _rank_label(result: CaseResult) -> str:
    return str(result.rank) if result.rank is not None else "miss"


def _matches_expected(hit: dict[str, Any], criterion: dict[str, str]) -> bool:
    for key, expected_value in criterion.items():
        if key.endswith("_contains"):
            field_name = key.removesuffix("_contains")
            actual_value = str(hit.get(field_name, "")).lower()
            if expected_value.lower() not in actual_value:
                return False
        elif hit.get(key) != expected_value:
            return False
    return True


def _success_rate(results: list[CaseResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.matched) / len(results)


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _optional_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=EVAL_MODES, default="rerank")
    parser.add_argument("--retrieval-limit", type=int, default=50)
    parser.add_argument("--reranking-limit", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
