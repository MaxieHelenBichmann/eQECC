"""CSV helpers shared by the extractors."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COLLECTED_DATA_DIR = ROOT / "paper" / "data" / "collected"
ALGORITHM_DATA_DIR = COLLECTED_DATA_DIR / "algorithms"
RESULTS_DIR = ROOT / "paper" / "results"

STATISTICS_FIELDS = (
    "algorithm", "n", "k", "positive", "seed", "nr_seeds", "mean_seconds",
    "stddev_seconds", "maximum_seconds", "num_cases", "num_successful",
    "num_unexpected", "num_timeouts", "num_memory_limited", "num_errors",
    "num_generation_errors",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        sys.exit(f"missing input {path.relative_to(ROOT)}: run the collector or extractor that writes it first")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def as_int(value) -> int:
    return int(value or 0)


def as_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def problem_for_algorithm(algorithm: str) -> str:
    for problem in ("pm_stb", "pm_css", "lc_stb"):
        if algorithm.startswith(f"{problem}_"):
            return problem
    raise ValueError(f"cannot infer problem family from algorithm {algorithm!r}")


def read_statistics(path: Path) -> list[dict[str, str]]:
    # the collectors append; the latest row per invocation key wins
    latest = {}
    for row in read_csv(path):
        latest[tuple(row[field] for field in ("algorithm", "n", "k", "positive", "seed"))] = row
    return list(latest.values())


def _pooled(values: Sequence[tuple[int, float, float]]) -> tuple[float | None, float | None]:
    total = sum(count for count, _, _ in values)
    if total == 0:
        return None, None
    average = sum(count * mean for count, mean, _ in values) / total
    if total == 1:
        return average, 0.0
    squared = sum(
        max(0, count - 1) * deviation**2 + count * (mean - average) ** 2
        for count, mean, deviation in values
    )
    return average, math.sqrt(squared / (total - 1))


def combine_statistic_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Pool the positive and negative statistics rows of one parameter cell."""
    distributions = []
    maxima = []
    for row in rows:
        # runtimes cover successful, unexpected, and timed-out calls; memory and execution failures are excluded
        observed = as_int(row["num_cases"]) - as_int(row["num_memory_limited"]) - as_int(row["num_errors"])
        average = as_float(row["mean_seconds"])
        maximum = as_float(row["maximum_seconds"])
        if observed and average is not None:
            distributions.append((observed, average, as_float(row["stddev_seconds"]) or 0.0))
        if maximum is not None:
            maxima.append(maximum)
    mean_seconds, stddev_seconds = _pooled(distributions)
    sample = rows[0]
    result = {
        "algorithm": sample["algorithm"],
        "problem": problem_for_algorithm(sample["algorithm"]),
        "n": as_int(sample["n"]),
        "k": as_int(sample["k"]),
        "r": as_int(sample["n"]) - as_int(sample["k"]),
        "num_requested": sum(as_int(row["nr_seeds"]) for row in rows),
        "num_cases": sum(as_int(row["num_cases"]) for row in rows),
        "num_observed": sum(count for count, _, _ in distributions),
        "num_successful": sum(as_int(row["num_successful"]) for row in rows),
        "num_unexpected": sum(as_int(row["num_unexpected"]) for row in rows),
        "num_timeouts": sum(as_int(row["num_timeouts"]) for row in rows),
        "num_memory_limited": sum(as_int(row["num_memory_limited"]) for row in rows),
        "num_errors": sum(as_int(row["num_errors"]) for row in rows),
        "num_generation_errors": sum(as_int(row["num_generation_errors"]) for row in rows),
        "mean_seconds": mean_seconds,
        "stddev_seconds": stddev_seconds,
        "maximum_seconds": max(maxima) if maxima else None,
        "has_positive": any(as_bool(row["positive"]) for row in rows),
        "has_negative": any(not as_bool(row["positive"]) for row in rows),
    }
    result["complete"] = (
        result["has_positive"]
        and result["has_negative"]
        and result["num_successful"] == result["num_requested"]
        and not (
            result["num_unexpected"] or result["num_timeouts"] or result["num_memory_limited"]
            or result["num_errors"] or result["num_generation_errors"]
        )
    )
    return result


def aggregate_statistics(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], as_int(row["n"]), as_int(row["k"]))].append(row)
    return [combine_statistic_rows(group) for _, group in sorted(grouped.items())]


def load_algorithm(algorithm: str, directory: Path = ALGORITHM_DATA_DIR) -> list[dict[str, str]]:
    return read_statistics(directory / f"{algorithm}.csv")
