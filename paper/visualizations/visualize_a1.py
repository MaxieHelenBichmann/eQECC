"""Rejection maps per invariant and an overall-rate table (A1)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

from paper.experiments.common import RESULTS_DIR, read_csv
from paper.visualizations.common import (
    COLOR_PAPER_BLUE, COLOR_PAPER_GRAY_DARK, COLOR_PAPER_GRAY_LIGHT, COLOR_PAPER_GRAY_VERY_DARK,
    COLOR_PAPER_GRAY_VERY_LIGHT, COLOR_PAPER_GREEN_RAMP, COLOR_PAPER_ORANGE_RAMP,
    COLOR_PAPER_PINK_RAMP, COLOR_PAPER_WHITE, WIDE_TEXT_SCALE,
    half_cell_key, outline_partition, parameter_axis, partition_cell, save_png, scalar_mappable, use_style,
)

INPUT = RESULTS_DIR / "a1" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a1" / "a1.png"

CMAPS = {
    "linear_dependency": LinearSegmentedColormap.from_list("linear_dependency", COLOR_PAPER_GREEN_RAMP),
    "signatures": LinearSegmentedColormap.from_list("signatures", COLOR_PAPER_PINK_RAMP),
    "local_invariant": LinearSegmentedColormap.from_list("local_invariant", COLOR_PAPER_ORANGE_RAMP),
}
LABELS = {"linear_dependency": "Linear column dependencies", "signatures": "Signatures", "local_invariant": "Local invariant"}


def _aggregate(rows, problem: str) -> dict[tuple[int, int, str], tuple[int, int]]:
    """(rejected, valid) per (n, r, invariant)."""
    values = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["problem"] == problem and row["invariant"] != "combined":
            key = (int(row["n"]), int(row["r"]), row["invariant"])
            values[key][0] += int(row["num_rejected"])
            values[key][1] += int(row["num_valid"])
    return {key: tuple(value) for key, value in values.items()}


def _draw(ax, families, invariant: str) -> None:
    for index, values in enumerate(families):
        for (n, r, name), (rejected, valid) in values.items():
            if name != invariant or valid <= 0:
                continue
            partition_cell(ax, n, r, index, len(families), CMAPS[invariant](rejected / valid))
            if rejected == 0:
                outline_partition(ax, n, r, index, len(families), COLOR_PAPER_GRAY_VERY_DARK)


def _overall(values, invariant: str) -> float | None:
    selected = [value for (*_, name), value in values.items() if name == invariant]
    rejected = sum(value[0] for value in selected)
    valid = sum(value[1] for value in selected)
    return 100 * rejected / valid if valid else None


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _render_overall_table(pm_stb, pm_css, lc, output: Path) -> Path:
    values = (
        _overall(pm_stb, "linear_dependency"),
        _overall(pm_css, "linear_dependency"),
        _overall(pm_stb, "signatures"),
        _overall(pm_css, "signatures"),
        _overall(lc, "local_invariant"),
    )
    subheaders = ("General stabilizer", "CSS", "General stabilizer", "CSS", "General stabilizer (LC)")
    body_height = 0.67

    figure, ax = plt.subplots(figsize=(7.2, 1.55))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.77)
    ax.axis("off")
    table = ax.table(
        cellText=[[_rate(value) for value in values]],
        colLabels=subheaders,
        colWidths=[0.2] * 5,
        cellLoc="center",
        colLoc="center",
        bbox=(0, 0, 1, body_height),
    )
    table.auto_set_font_size(False)
    for column in range(5):
        header = table[(0, column)]
        header.set_facecolor(COLOR_PAPER_GRAY_LIGHT)
        header.set_edgecolor(COLOR_PAPER_WHITE)
        header.get_text().set_color("#202020")
        header.get_text().set_fontsize(7.5)
        header.get_text().set_fontweight("bold")
        cell = table[(1, column)]
        cell.set_facecolor(COLOR_PAPER_GRAY_VERY_LIGHT)
        cell.set_edgecolor(COLOR_PAPER_WHITE)
        cell.get_text().set_fontsize(10)
        if cell.get_text().get_text() != "—":
            cell.get_text().set_color(COLOR_PAPER_BLUE)
            cell.get_text().set_fontweight("bold")

    major_headers = (
        (0.0, 0.4, LABELS["linear_dependency"]),
        (0.4, 0.4, LABELS["signatures"]),
        (0.8, 0.2, LABELS["local_invariant"]),
    )
    for x, width, label in major_headers:
        ax.add_patch(Rectangle(
            (x, body_height), width, 1 - body_height, transform=ax.transAxes,
            facecolor=COLOR_PAPER_GRAY_DARK, edgecolor=COLOR_PAPER_WHITE, linewidth=1.0,
        ))
        ax.text(
            x + width / 2, body_height + (1 - body_height) / 2, label, transform=ax.transAxes,
            ha="center", va="center", color="#202020", fontsize=8, fontweight="bold",
        )
    figure.suptitle("Overall rejection rates", fontsize=11, fontweight="bold", y=0.96)
    return save_png(figure, output)


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    rows = read_csv(input_file)
    pm_stb = _aggregate(rows, "pm_stb")
    pm_css = _aggregate(rows, "pm_css")
    lc = _aggregate(rows, "lc_stb")

    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.7))
    figure.subplots_adjust(left=0.055, right=0.90, bottom=0.22, top=0.80, wspace=0.18)
    parameter_axis(axes[0], LABELS["linear_dependency"])
    parameter_axis(axes[1], LABELS["signatures"])
    parameter_axis(axes[2], "Local Clifford Equivalence")
    _draw(axes[0], (pm_stb, pm_css), "linear_dependency")
    _draw(axes[1], (pm_stb, pm_css), "signatures")
    _draw(axes[2], (lc,), "local_invariant")

    handles = [
        half_cell_key("left", COLOR_PAPER_GRAY_DARK, f"Stabilizer: {_rate(_overall(pm_stb, 'linear_dependency'))} linear, {_rate(_overall(pm_stb, 'signatures'))} signatures"),
        half_cell_key("right", COLOR_PAPER_GRAY_DARK, f"CSS: {_rate(_overall(pm_css, 'linear_dependency'))} linear, {_rate(_overall(pm_css, 'signatures'))} signatures"),
        Patch(facecolor=CMAPS["local_invariant"](0.72), label=f"{LABELS['local_invariant']} ({_rate(_overall(lc, 'local_invariant'))} overall)"),
        Patch(facecolor=CMAPS["linear_dependency"](0), edgecolor=COLOR_PAPER_GRAY_VERY_DARK, label="0% rejected (measured)"),
        Patch(facecolor=COLOR_PAPER_GRAY_VERY_LIGHT, edgecolor="none", label="not measured"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=9)
    figure.suptitle("Invariants' Rejection Patterns and Rates of Inequivalent Codes", fontsize=12 * WIDE_TEXT_SCALE, y=0.96)

    bar = figure.colorbar(scalar_mappable("Greys", Normalize(0, 1)), ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("Deeper color means\nmore rejected instances")
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    main_output = save_png(figure, output)
    _render_overall_table(pm_stb, pm_css, lc, output.with_name(f"{output.stem}_overall{output.suffix}"))
    return main_output


if __name__ == "__main__":
    render()
