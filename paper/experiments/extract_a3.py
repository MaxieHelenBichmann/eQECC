"""Invariant runtime relative to the best backend of the same cell (A3)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from paper.experiments.common import (
    ALGORITHM_DATA_DIR, COLLECTED_DATA_DIR, RESULTS_DIR,
    aggregate_statistics, as_bool, as_float, load_algorithm, read_csv, write_csv,
)
from paper.experiments.extract_a5 import A5_ALGORITHMS, select_winners

INVARIANT_INPUT = COLLECTED_DATA_DIR / "invariant_timings.csv"
OUTPUT = RESULTS_DIR / "a3" / "by_cell.csv"
FIELDS = (
    "problem", "invariant", "n", "k", "r", "invariant_mean_seconds",
    "invariant_stddev_seconds", "backend_algorithm", "backend_mean_seconds",
    "backend_selection", "backend_num_timeouts", "relative_runtime",
    "num_invariant_requested", "num_invariant_successful",
)
SEEDS_PER_POLARITY = 5


def read_invariant_cells(path: Path) -> list[dict]:
    # latest row per key wins; a cell needs all 5 + 5 runs to have succeeded
    latest = {}
    for row in read_csv(path):
        latest[tuple(row[field] for field in ("problem", "invariant", "n", "k", "positive", "seed"))] = row
    grouped = defaultdict(list)
    for row in latest.values():
        grouped[(row["problem"], row["invariant"], int(row["n"]), int(row["k"]))].append(row)

    cells = []
    for (problem, invariant, n, k), group in sorted(grouped.items()):
        positives = sum(as_bool(row["positive"]) for row in group)
        runtimes = [as_float(row["runtime_seconds"]) for row in group if row["status"] == "success"]
        if positives != SEEDS_PER_POLARITY or len(runtimes) != 2 * SEEDS_PER_POLARITY or None in runtimes:
            continue
        cells.append({
            "problem": problem, "invariant": invariant, "n": n, "k": k, "r": n - k,
            "mean_seconds": mean(runtimes), "stddev_seconds": stdev(runtimes),
            "num_requested": len(group), "num_successful": len(runtimes),
        })
    return cells


def extract(
    invariant_input: Path = INVARIANT_INPUT,
    algorithm_directory: Path = ALGORITHM_DATA_DIR,
    output_file: Path = OUTPUT,
    algorithm_names=A5_ALGORITHMS,
) -> list[dict]:
    algorithm_rows = [row for algorithm in algorithm_names for row in load_algorithm(algorithm, algorithm_directory)]
    # the baseline is A5's winner, including timeout fallbacks
    backends = {
        (winner["problem"], winner["n"], winner["k"]): winner
        for winner in select_winners(aggregate_statistics(algorithm_rows))
    }
    output = []
    for cell in read_invariant_cells(invariant_input):
        backend = backends.get((cell["problem"], cell["n"], cell["k"]))
        if backend is None or not backend["mean_seconds"]:
            continue
        output.append({
            "problem": cell["problem"], "invariant": cell["invariant"],
            "n": cell["n"], "k": cell["k"], "r": cell["r"],
            "invariant_mean_seconds": cell["mean_seconds"],
            "invariant_stddev_seconds": cell["stddev_seconds"],
            "backend_algorithm": backend["winner"],
            "backend_mean_seconds": backend["mean_seconds"],
            "backend_selection": backend["selection"],
            "backend_num_timeouts": backend["winner_num_timeouts"],
            "relative_runtime": cell["mean_seconds"] / backend["mean_seconds"],
            "num_invariant_requested": cell["num_requested"],
            "num_invariant_successful": cell["num_successful"],
        })
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
