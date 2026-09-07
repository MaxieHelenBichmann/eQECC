"""Invariant-to-backend runtime ratio maps (A3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paper.experiments.common import RESULTS_DIR, as_float, read_csv
from paper.visualizations.common import (
    COLOR_PAPER_DARK_BLUE,
    COLOR_PAPER_DARK_RED,
    COLOR_PAPER_GRAY_LIGHT,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    COLOR_PAPER_WHITE,
    RELATIVE_CMAP,
    WIDE_TEXT_SCALE,
    half_cell_key,
    outline_partition,
    parameter_axis,
    partition_cell,
    ratio_ticks,
    relative_norm,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a3" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a3" / "a3.png"

PANELS = (
    ("linear_dependency", "Linear Column Dependencies", ("pm_stb", "pm_css")),
    ("signatures", "Signatures", ("pm_stb", "pm_css")),
    ("local_invariant", "Local Invariant", ("lc_stb",)),
)


def draw_panel(ax, rows, problems, norm) -> None:
    for index, problem in enumerate(problems):
        for row in rows:
            ratio = as_float(row["relative_runtime"])
            if row["problem"] != problem or not ratio or ratio <= 0:
                continue
            n, r = int(row["n"]), int(row["r"])
            partition_cell(ax, n, r, index, len(problems), RELATIVE_CMAP(norm(ratio)))
            if row["backend_selection"] == "timeout_fallback":
                outline_partition(ax, n, r, index, len(problems), COLOR_PAPER_GRAY_VERY_VERY_DARK)


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    rows = read_csv(input_file)
    norm = relative_norm()

    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.7))
    figure.subplots_adjust(left=0.055, right=0.90, bottom=0.22, top=0.80, wspace=0.18)
    for ax, (invariant, title, problems) in zip(axes, PANELS):
        parameter_axis(ax, title, empty_color=COLOR_PAPER_GRAY_LIGHT)
        draw_panel(ax, [row for row in rows if row["invariant"] == invariant], problems, norm)

    figure.suptitle("Invariant Cost Relative to the Best-Performing Backend", fontsize=12 * WIDE_TEXT_SCALE, y=0.96)
    gray = COLOR_PAPER_GRAY_VERY_VERY_DARK
    figure.legend(
        handles=[
            half_cell_key("left", gray, "Stabilizer codes", size=8),
            half_cell_key("right", gray, "CSS codes", size=8),
            Patch(facecolor=COLOR_PAPER_DARK_BLUE, edgecolor="none", label="Invariant is cheaper"),
            Patch(facecolor=COLOR_PAPER_DARK_RED, edgecolor="none", label="Invariant costs more"),
            Patch(facecolor=COLOR_PAPER_WHITE, edgecolor=gray, linewidth=0.9, label="Backend timed out"),
        ],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.015),
    )
    bar = figure.colorbar(scalar_mappable(RELATIVE_CMAP, norm), ax=axes, fraction=0.025, pad=0.02, extend="both")
    ratio_ticks(bar)
    bar.set_label("$T_{\\mathrm{invariant}} \\,/\\, T_{\\mathrm{backend}}$")
    return save_png(figure, output)


if __name__ == "__main__":
    render()
