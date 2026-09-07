"""Fastest-algorithm maps per problem (A5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Polygon

from paper.experiments.common import RESULTS_DIR, read_csv
from paper.visualizations.common import (
    COLOR_PAPER_CYAN_STRONG, COLOR_PAPER_DARK_CYAN, COLOR_PAPER_DARK_PINK, COLOR_PAPER_DARK_RED,
    COLOR_PAPER_GRAY_VERY_DARK, COLOR_PAPER_GRAY_VERY_VERY_DARK, COLOR_PAPER_GREEN_DEEP,
    COLOR_PAPER_LIGHT_RED, COLOR_PAPER_LILA, COLOR_PAPER_ZX_BLUE, WIDE_TEXT_SCALE,
    parameter_axis, partition_cell, save_png, use_style,
)

INPUT = RESULTS_DIR / "a5" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a5" / "a5.png"
NEAR_TIE_RATIO = 1.05

PANELS = (
    ("pm_stb", "Permutation Equivalence\nfor Stabilizer Codes"),
    ("pm_css", "Permutation Equivalence\nfor CSS Codes"),
    ("lc_stb", "Local Clifford Equivalence\nfor Stabilizer Codes"),
)
METHODS = (
    ("sat", "SAT", COLOR_PAPER_DARK_CYAN),
    ("lse", "Graph-State LSE", COLOR_PAPER_DARK_PINK),
    ("graph_iso", "Graph Isomorphism", COLOR_PAPER_GREEN_DEEP),
    ("matroid", "Matroid Isomorphism", COLOR_PAPER_LIGHT_RED),
    ("bruteforce", "Brute force", COLOR_PAPER_DARK_RED),
    ("kls", "KLS Orbit", COLOR_PAPER_ZX_BLUE),
    ("classical", "Classical Approaches", COLOR_PAPER_LILA),
    ("aut", "Automorphism Group", COLOR_PAPER_CYAN_STRONG),
)


def method(algorithm: str) -> tuple[str, str]:
    for suffix, label, color in METHODS:
        if algorithm.endswith(f"_{suffix}"):
            return label, color
    return algorithm, COLOR_PAPER_GRAY_VERY_VERY_DARK


def overlay_runner_up(ax, n: int, r: int, color: str) -> None:
    # lower-right triangle; the winner keeps the upper-left one
    x, y = n - 0.5, r - 0.5
    ax.add_patch(Polygon([(x, y), (x + 1, y), (x + 1, y + 1)], closed=True, facecolor=color, edgecolor="none", zorder=3))


class SplitCellHandler(HandlerBase):
    def create_artists(self, legend, handle, xdescent, ydescent, width, height, fontsize, trans):
        x, y = -xdescent, -ydescent
        top = Polygon([(x, y), (x, y + height), (x + width, y + height)], closed=True,
                      facecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK, edgecolor="none", transform=trans)
        bottom = Polygon([(x, y), (x + width, y), (x + width, y + height)], closed=True,
                         facecolor=COLOR_PAPER_GRAY_VERY_DARK, edgecolor="none", transform=trans)
        return [top, bottom]


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    rows = read_csv(input_file)

    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.7))
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.22, top=0.80, wspace=0.18)
    for ax, (problem, title) in zip(axes, PANELS):
        parameter_axis(ax, title)
        for row in rows:
            if row["problem"] != problem:
                continue
            n, r = int(row["n"]), int(row["r"])
            partition_cell(ax, n, r, 0, 1, method(row["winner"])[1])
            ratio = float(row["speed_ratio"]) if row["speed_ratio"] else None
            if row["selection"] == "completed" and row["runner_up"] and ratio is not None and ratio <= NEAR_TIE_RATIO:
                overlay_runner_up(ax, n, r, method(row["runner_up"])[1])

    figure.suptitle("Best-Performing Prototype per Parameter Setting", fontsize=12 * WIDE_TEXT_SCALE, y=0.96)
    exclusions = sorted({
        item.strip() for row in rows for item in row.get("excluded_algorithms", "").split(";") if item.strip()
    })
    if exclusions:
        figure.text(0.5, 0.84, "Excluded for missing data or errors: " + ", ".join(exclusions), ha="center", va="center", fontsize=8)

    algorithms = {row["winner"] for row in rows} | {row["runner_up"] for row in rows if row["runner_up"]}
    present = {method(algorithm)[0] for algorithm in algorithms}
    split_key = Patch(label="top/bottom: within 5%")
    handles = [Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in METHODS if label in present]
    figure.legend(
        handles=[*handles, split_key], handler_map={split_key: SplitCellHandler()},
        loc="lower center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.025),
    )
    return save_png(figure, output)


if __name__ == "__main__":
    render()
