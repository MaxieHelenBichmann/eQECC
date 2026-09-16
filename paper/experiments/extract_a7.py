"""Median decision counts per X/Z rank split, and for clean vs row-mixed CSS tableaus (A7)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median

from paper.experiments.common import COLLECTED_DATA_DIR, RESULTS_DIR, as_bool, as_int, read_csv, write_csv

INPUT = COLLECTED_DATA_DIR / "sat_css_weakness.csv"
OUTPUT = RESULTS_DIR / "a7" / "rank_sweep.csv"
ROW_MIXING_OUTPUT = RESULTS_DIR / "a7" / "row_mixing.csv"

EXPERIMENT1 = "rank_sweep"
EXPERIMENT2 = "row_mixing"
ROW_MIXING_CONDITIONS = ("clean", "mixed")

CONDITION_ORDER = ("css", "general")
RX_LABELS = ("0", "1", "2", "r/2", "r-2", "r-1", "r")
KEY_FIELDS = ("experiment", "condition", "rx", "sample", "seed", "n", "k", "measurement", "probe")
FIELDS = (
    "n",
    "k",
    "r",
    "condition",
    "rx",
    "rz",
    "rx_label",
    "runs",
    "completed",
    "timeouts",
    "median_decisions",
    "row_operation_variables",
)
ROW_MIXING_FIELDS = ("n", "k", "r", "condition", "runs", "completed", "timeouts", "median_decisions")


def _median(values):
    return median(values) if values else ""


def _rx(row) -> int | str:
    return as_int(row["rx"]) if row["rx"].strip() else ""


def rx_label(r: int, rx: int | str) -> str:
    if rx == "":
        return "general"
    return dict(zip((0, 1, 2, r // 2, r - 2, r - 1, r), RX_LABELS, strict=True))[rx]


def _sort_key(item):
    (n, condition, rx), _ = item
    return (n, CONDITION_ORDER.index(condition), rx if rx != "" else -1)


def _rank_sweep(rows) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["experiment"] == EXPERIMENT1:
            groups[(as_int(row["n"]), row["condition"], _rx(row))].append(row)

    output = []
    for (n, condition, rx), cell in sorted(groups.items(), key=_sort_key):
        r = as_int(cell[0]["r"])
        completed = [row for row in cell if row["result"] == "sat"]
        output.append(
            {
                "n": n,
                "k": as_int(cell[0]["k"]),
                "r": r,
                "condition": condition,
                "rx": rx,
                "rz": r - rx if rx != "" else "",
                "rx_label": rx_label(r, rx),
                "runs": len(cell),
                "completed": len(completed),
                "timeouts": sum(as_bool(row["timed_out"]) for row in cell),
                "median_decisions": _median([as_int(row["decisions"]) for row in completed]),
                "row_operation_variables": as_int(cell[0]["row_operation_variables"]),
            }
        )
    return output


def _row_mixing(rows) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["experiment"] == EXPERIMENT2:
            groups[(as_int(row["n"]), row["condition"])].append(row)

    output = []
    for (n, condition), cell in sorted(
        groups.items(), key=lambda item: (item[0][0], ROW_MIXING_CONDITIONS.index(item[0][1]))
    ):
        completed = [row for row in cell if row["result"] == "sat"]
        output.append(
            {
                "n": n,
                "k": as_int(cell[0]["k"]),
                "r": as_int(cell[0]["r"]),
                "condition": condition,
                "runs": len(cell),
                "completed": len(completed),
                "timeouts": sum(as_bool(row["timed_out"]) for row in cell),
                "median_decisions": _median([as_int(row["decisions"]) for row in completed]),
            }
        )
    return output


def extract(
    input_file: Path = INPUT, output_file: Path = OUTPUT, row_mixing_output: Path = ROW_MIXING_OUTPUT
) -> tuple[list[dict], list[dict]]:
    latest = {}
    for row in read_csv(input_file):
        latest[tuple(row[field] for field in KEY_FIELDS)] = row
    sweep = _rank_sweep(latest.values())
    row_mixing = _row_mixing(latest.values())
    write_csv(output_file, sweep, FIELDS)
    write_csv(row_mixing_output, row_mixing, ROW_MIXING_FIELDS)
    return sweep, row_mixing


if __name__ == "__main__":
    extract()
