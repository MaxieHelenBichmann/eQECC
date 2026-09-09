"""Mean pairwise refinement of the signature partitions per cell (A2)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from paper.experiments.common import COLLECTED_DATA_DIR, RESULTS_DIR, read_csv, write_csv

INPUT = COLLECTED_DATA_DIR / "signature_space.csv"
OUTPUT = RESULTS_DIR / "a2" / "by_cell.csv"
PROBLEM_FIELDS = ("problem", "num_cells", "num_requested", "num_valid", "num_censored", "mean_pairwise_refinement")
FIELDS = (
    "problem",
    "n",
    "k",
    "r",
    "num_requested",
    "num_valid",
    "mean_pairwise_refinement",
    "stddev_pairwise_refinement",
    "num_censored",
)


def pairwise_refinement(q_pairs: float, n: int) -> float:
    """Normalize fraction of distinct qubit pairs separated by the signature: 0 for one class, 1 for all singletons."""
    return (1 - q_pairs) / (1 - 1 / n)


def overall(cells: list[dict]) -> list[dict]:
    """Instance-weighted mean over every valid seed of a problem, across all parameter settings."""
    rows = []
    for problem in sorted({cell["problem"] for cell in cells}):
        group = [cell for cell in cells if cell["problem"] == problem]
        valid = sum(int(cell["num_valid"]) for cell in group)
        weighted = sum(
            float(cell["mean_pairwise_refinement"]) * int(cell["num_valid"]) for cell in group if int(cell["num_valid"])
        )
        rows.append(
            {
                "problem": problem,
                "num_cells": len(group),
                "num_requested": sum(int(cell["num_requested"]) for cell in group),
                "num_valid": valid,
                "num_censored": sum(int(cell["num_censored"]) for cell in group),
                "mean_pairwise_refinement": weighted / valid if valid else "",
            }
        )
    return rows


def extract(input_file: Path = INPUT, output_file: Path = OUTPUT) -> list[dict]:
    groups = defaultdict(list)
    for row in read_csv(input_file):
        groups[(row["problem"], int(row["n"]), int(row["k"]))].append(row)
    cells = []
    for (problem, n, k), group in sorted(groups.items()):
        values = [pairwise_refinement(float(row["q_pairs"]), n) for row in group if row["status"] == "success"]
        cells.append(
            {
                "problem": problem,
                "n": n,
                "k": k,
                "r": n - k,
                "num_requested": len(group),
                "num_valid": len(values),
                "mean_pairwise_refinement": mean(values) if values else "",
                "stddev_pairwise_refinement": stdev(values) if len(values) > 1 else (0.0 if values else ""),
                "num_censored": len(group) - len(values),
            }
        )
    write_csv(output_file, cells, FIELDS)
    write_csv(output_file.with_name("by_problem.csv"), overall(cells), PROBLEM_FIELDS)
    return cells


if __name__ == "__main__":
    extract()
