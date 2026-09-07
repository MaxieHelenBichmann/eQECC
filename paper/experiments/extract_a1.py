"""Rejection counts per cell and overall (A1)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from paper.experiments.common import COLLECTED_DATA_DIR, RESULTS_DIR, as_bool, read_csv, write_csv

INPUT = COLLECTED_DATA_DIR / "invariant_rejections.csv"
OUTPUT_DIRECTORY = RESULTS_DIR / "a1"
CELL_FIELDS = (
    "problem",
    "n",
    "k",
    "r",
    "invariant",
    "num_requested",
    "num_valid",
    "num_rejected",
    "rejection_percentage",
    "num_censored",
)
OVERALL_FIELDS = ("problem", "invariant", "num_valid", "num_rejected", "rejection_percentage")


def extract(input_file: Path = INPUT, output_directory: Path = OUTPUT_DIRECTORY) -> list[dict]:
    rows = read_csv(input_file)
    # an instance counts as rejected by "combined" if any of its invariants rejected it
    by_instance = defaultdict(list)
    for row in rows:
        by_instance[(row["problem"], row["instance_id"])].append(row)
    for group in by_instance.values():
        valid = all(row["status"] == "success" for row in group)
        rows.append(
            {
                **group[0],
                "invariant": "combined",
                "status": "success" if valid else "censored",
                "rejected": str(any(as_bool(row["rejected"]) for row in group)) if valid else "",
            }
        )

    cells = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["problem"], int(row["n"]), int(row["k"]), row["invariant"])].append(row)
    for (problem, n, k, invariant), group in sorted(groups.items()):
        valid_rows = [row for row in group if row["status"] == "success"]
        rejected = sum(as_bool(row["rejected"]) for row in valid_rows)
        cells.append(
            {
                "problem": problem,
                "n": n,
                "k": k,
                "r": n - k,
                "invariant": invariant,
                "num_requested": len(group),
                "num_valid": len(valid_rows),
                "num_rejected": rejected,
                "rejection_percentage": 100 * rejected / len(valid_rows) if valid_rows else "",
                "num_censored": len(group) - len(valid_rows),
            }
        )

    overall = []
    totals = defaultdict(list)
    for row in rows:
        totals[(row["problem"], row["invariant"])].append(row)
    for (problem, invariant), group in sorted(totals.items()):
        valid_rows = [row for row in group if row["status"] == "success"]
        rejected = sum(as_bool(row["rejected"]) for row in valid_rows)
        overall.append(
            {
                "problem": problem,
                "invariant": invariant,
                "num_valid": len(valid_rows),
                "num_rejected": rejected,
                "rejection_percentage": 100 * rejected / len(valid_rows) if valid_rows else "",
            }
        )

    write_csv(output_directory / "by_cell.csv", cells, CELL_FIELDS)
    write_csv(output_directory / "overall.csv", overall, OVERALL_FIELDS)
    return cells


if __name__ == "__main__":
    extract()
