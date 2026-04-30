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


def main() -> int:
    args = _parse_args()
    cases = load_cases(args.dataset)
    if not cases:
        print(f"No eval cases found in {args.dataset}", file=sys.stderr)
        return 2

    results = run_eval(
        cases=cases,
        base_url=args.base_url,
        k=args.k,
        timeout_seconds=args.timeout,
    )
    print_report(results, k=args.k)
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
    k: int,
    timeout_seconds: float,
) -> list[CaseResult]:
    """Run all retrieval cases against a live /search endpoint."""
    results: list[CaseResult] = []
    search_url = f"{base_url.rstrip('/')}/search"
    with httpx.Client(timeout=timeout_seconds) as client:
        for case in cases:
            started = perf_counter()
            try:
                response = client.post(search_url, json={"query": case.question, "k": k})
                response.raise_for_status()
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


def first_matching_rank(
    hits: list[dict[str, Any]],
    expected: list[dict[str, str]],
) -> Optional[int]:
    """Return the 1-based rank of the first hit matching any expected criterion."""
    for index, hit in enumerate(hits, start=1):
        if any(_matches_expected(hit, criterion) for criterion in expected):
            return index
    return None


def print_report(results: list[CaseResult], *, k: int) -> None:
    """Print aggregate retrieval metrics and weak cases."""
    total = len(results)
    matched = sum(1 for result in results if result.matched)
    mrr = sum(1 / result.rank for result in results if result.rank is not None) / total
    average_latency = sum(result.elapsed_ms for result in results) / total

    print(f"cases: {total}")
    print(f"hit_rate@{k}: {matched / total:.3f} ({matched}/{total})")
    print(f"mrr@{k}: {mrr:.3f}")
    print(f"avg_latency_ms: {average_latency:.1f}")

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
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
