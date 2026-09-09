"""Signature refinement maps for stabilizer and CSS codes (A2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from paper.experiments.common import RESULTS_DIR, read_csv
from paper.experiments.extract_a2 import overall
from paper.visualizations.common import (
    COLOR_PAPER_GRAY_VERY_LIGHT,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    SIGNATURE_CMAP,
    aggregate_cells,
    outline_partition,
    parameter_axis,
    partition_cell,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a2" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a2" / "a2.png"
PANELS = (("pm_stb", "Stabilizer-Code Signatures"), ("pm_css", "CSS-Code Signatures"))


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    rows = read_csv(input_file)
    norm = Normalize(vmin=0.0, vmax=1.0)

    use_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.5))
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.17, top=0.84, wspace=0.18)
    for ax, (problem, title) in zip(axes, PANELS):
        parameter_axis(ax, title)
        problem_rows = [row for row in rows if row["problem"] == problem]
        cells = aggregate_cells(problem_rows, "mean_pairwise_refinement", "num_valid")
        censored = {(int(row["n"]), int(row["r"])) for row in problem_rows if int(row["num_censored"])}
        for (n, r), cell in cells.items():
            if int(cell["num_successful"]):
                partition_cell(ax, n, r, 0, 1, SIGNATURE_CMAP(norm(float(cell["mean_value"]))))
            elif (n, r) in censored:
                outline_partition(ax, n, r, 0, 1, COLOR_PAPER_GRAY_VERY_VERY_DARK)
        (summary,) = overall(problem_rows)
        if summary["mean_pairwise_refinement"] != "":
            ax.text(
                0.04,
                0.96,
                f"Mean over all instances: {summary['mean_pairwise_refinement']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#202020",
            )
    figure.suptitle("Pairwise Refinement Induced by Permutation Signatures", fontsize=12)
    figure.legend(
        handles=[
            Patch(
                facecolor=COLOR_PAPER_GRAY_VERY_LIGHT,
                edgecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK,
                linewidth=0.9,
                label="All instances timed out, no mean recoverable",
            )
        ],
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.47, 0.015),
    )
    bar = figure.colorbar(scalar_mappable(SIGNATURE_CMAP, norm), ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("Fraction of qubit pairs distinguished by signature\n(0 = no refinement, 1 = complete refinement)")
    return save_png(figure, output)


if __name__ == "__main__":
    render()
