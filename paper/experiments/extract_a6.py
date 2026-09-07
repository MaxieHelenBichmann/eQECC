"""SAT runtimes with both encodings on both code families (A6)."""

from __future__ import annotations

import math
from pathlib import Path

from paper.experiments.common import (
    ALGORITHM_DATA_DIR,
    COLLECTED_DATA_DIR,
    RESULTS_DIR,
    aggregate_statistics,
    load_algorithm,
    read_statistics,
    write_csv,
)

EXTRA_INPUT = COLLECTED_DATA_DIR / "pm_stb_sat_on_css.csv"
OUTPUT = RESULTS_DIR / "a6" / "by_cell.csv"
NMAX = 25
TIMEOUT_SECONDS = 5_400.0
VARIANTS = (
    ("pm_stb_sat_on_stabilizer", "pm_stb_sat", "stabilizer"),
    ("pm_css_sat_on_css", "pm_css_sat", "css"),
    ("pm_stb_sat_on_css", "pm_stb_sat_on_css", "css"),
)
FIELDS = (
    "variant",
    "algorithm",
    "code_family",
    "n",
    "k",
    "r",
    "num_requested",
    "num_successful",
    "mean_seconds",
    "stddev_seconds",
    "maximum_seconds",
    "hx_hz_log_scale_improvement_percentage",
    "num_timeouts",
    "num_memory_limited",
    "num_errors",
    "num_unexpected",
    "num_generation_errors",
)


def extract(
    algorithm_directory: Path = ALGORITHM_DATA_DIR,
    extra_input: Path = EXTRA_INPUT,
    output_file: Path = OUTPUT,
) -> list[dict]:
    aggregated = {
        "pm_stb_sat": aggregate_statistics(load_algorithm("pm_stb_sat", algorithm_directory)),
        "pm_css_sat": aggregate_statistics(load_algorithm("pm_css_sat", algorithm_directory)),
        "pm_stb_sat_on_css": aggregate_statistics(read_statistics(extra_input)),
    }
    output = [
        {**cell, "variant": variant, "code_family": family, "hx_hz_log_scale_improvement_percentage": ""}
        for variant, algorithm, family in VARIANTS
        for cell in aggregated[algorithm]
    ]

    # improvement of the check-matrix over the tableau encoding as a share of the figure's log runtime range
    displayed_means = [
        row["mean_seconds"] for row in output if row["n"] <= NMAX and row["mean_seconds"] and row["num_successful"]
    ]
    log_span = math.log(max(max(displayed_means), TIMEOUT_SECONDS) / min(displayed_means))
    css_means = {
        (row["n"], row["k"]): row["mean_seconds"]
        for row in output
        if row["variant"] == "pm_css_sat_on_css" and row["mean_seconds"]
    }
    for row in output:
        css_mean = css_means.get((row["n"], row["k"]))
        if row["variant"] == "pm_stb_sat_on_css" and css_mean and row["mean_seconds"]:
            row["hx_hz_log_scale_improvement_percentage"] = (
                100.0 * (math.log(row["mean_seconds"]) - math.log(css_mean)) / log_span
            )
    write_csv(output_file, output, FIELDS)
    return output


if __name__ == "__main__":
    extract()
