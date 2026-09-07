"""Runtime statistics of the graph and matroid representation algorithms (A4)."""

from __future__ import annotations

from pathlib import Path

from paper.experiments.common import ALGORITHM_DATA_DIR, RESULTS_DIR, aggregate_statistics, load_algorithm, write_csv

ALGORITHMS = ("pm_stb_graph_iso", "pm_css_matroid", "lc_stb_graph_iso")
OUTPUT = RESULTS_DIR / "a4" / "by_cell.csv"
FIELDS = (
    "problem", "algorithm", "n", "k", "r", "num_requested", "num_successful",
    "mean_seconds", "stddev_seconds", "maximum_seconds",
    "num_timeouts", "num_memory_limited", "num_errors", "num_unexpected", "num_generation_errors",
)


def extract(algorithm_directory: Path = ALGORITHM_DATA_DIR, output_file: Path = OUTPUT) -> list[dict]:
    rows = [row for algorithm in ALGORITHMS for row in load_algorithm(algorithm, algorithm_directory)]
    cells = aggregate_statistics(rows)
    write_csv(output_file, cells, FIELDS)
    return cells


if __name__ == "__main__":
    extract()
