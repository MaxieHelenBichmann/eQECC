"""Median decision counts per condition (A7)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median

from paper.experiments.common import COLLECTED_DATA_DIR, RESULTS_DIR, as_bool, as_float, as_int, read_csv, write_csv

INPUT = COLLECTED_DATA_DIR / "a7_sat_css_structure.csv"
OUTPUT = RESULTS_DIR / "a7" / "by_cell.csv"

EXPERIMENT1 = "permutation_survival"
EXPERIMENT2 = "row_mixing"
CONDITION_ORDER = ("A", "B1", "B2", "C", "clean", "mixed")
KEY_FIELDS = ("experiment", "condition", "sample", "seed", "n", "k", "measurement", "probe")
FIELDS = (
    "experiment", "condition", "n", "k", "r", "rx", "rz",
    "base_runs", "base_completed", "base_timeouts", "median_base_decisions", "median_base_seconds",
    "invalid_mapping_attempts", "invalid_mappings_proven", "invalid_mapping_automorphisms",
    "invalid_mapping_timeouts", "median_invalid_mapping_decisions", "median_invalid_mapping_seconds",
)


def _median(values):
    return median(values) if values else ""


def extract(input_file: Path = INPUT, output_file: Path = OUTPUT) -> list[dict]:
    latest = {}
    for row in read_csv(input_file):
        latest[tuple(row[field] for field in KEY_FIELDS)] = row
    groups = defaultdict(list)
    for row in latest.values():
        groups[(row["experiment"], row["condition"], as_int(row["n"]), as_int(row["k"]))].append(row)

    output = []
    for (experiment, condition, n, k), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2], CONDITION_ORDER.index(item[0][1]))):
        base = [row for row in rows if row["measurement"] == "base"]
        completed_base = [row for row in base if row["result"] == "sat"]
        mappings = [row for row in rows if row["measurement"] == "invalid_mapping"]
        proven = [row for row in mappings if row["result"] == "unsat"]
        output.append({
            "experiment": experiment, "condition": condition, "n": n, "k": k,
            "r": as_int(rows[0]["r"]), "rx": as_int(rows[0]["rx"]), "rz": as_int(rows[0]["rz"]),
            "base_runs": len(base),
            "base_completed": len(completed_base),
            "base_timeouts": sum(as_bool(row["timed_out"]) for row in base),
            "median_base_decisions": _median([as_int(row["decisions"]) for row in completed_base]),
            "median_base_seconds": _median([as_float(row["solve_seconds"]) or 0.0 for row in completed_base]),
            "invalid_mapping_attempts": len(mappings),
            "invalid_mappings_proven": len(proven),
            "invalid_mapping_automorphisms": sum(row["result"] == "sat" for row in mappings),
            "invalid_mapping_timeouts": sum(as_bool(row["timed_out"]) for row in mappings),
            "median_invalid_mapping_decisions": _median([as_int(row["decisions"]) for row in proven]),
            "median_invalid_mapping_seconds": _median([as_float(row["solve_seconds"]) or 0.0 for row in proven]),
        })
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
