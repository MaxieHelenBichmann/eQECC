"""Rejection maps per invariant and an overall-rate table (A1)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

from paper.experiments.common import RESULTS_DIR, read_csv
from paper.visualizations.common import (
    COLOR_PAPER_GRAY_DARK,
    COLOR_PAPER_GRAY_VERY_DARK,
    COLOR_PAPER_GREEN_RAMP,
    COLOR_PAPER_ORANGE_RAMP,
    COLOR_PAPER_PINK_RAMP,
    COLOR_PAPER_WHITE,
    WIDE_TEXT_SCALE,
    half_cell_key,
    outline_partition,
    parameter_axis,
    partition_cell,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a1" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a1" / "a1.png"

CMAPS = {
    name: LinearSegmentedColormap.from_list(name, (COLOR_PAPER_WHITE, *ramp[1:]))
    for name, ramp in (
        ("linear_dependency", COLOR_PAPER_GREEN_RAMP),
        ("signatures", COLOR_PAPER_PINK_RAMP),
        ("local_invariant", COLOR_PAPER_ORANGE_RAMP),
    )
}
LABELS = {
    "linear_dependency": "Linear column dependencies",
    "signatures": "Signatures",
    "local_invariant": "Local invariant",
}


def _aggregate(rows, problem: str) -> dict[tuple[int, int, str], tuple[int, int]]:
    """(rejected, valid) per (n, r, invariant)."""
    values: defaultdict[tuple[int, int, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["problem"] == problem and row["invariant"] != "combined":
            key = (int(row["n"]), int(row["r"]), row["invariant"])
            values[key][0] += int(row["num_rejected"])
            values[key][1] += int(row["num_valid"])
    return {key: (value[0], value[1]) for key, value in values.items()}


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


def _annotate(ax, lines: list[str]) -> None:
    ax.text(
        0.04,
        0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8 * WIDE_TEXT_SCALE,
        color="#202020",
    )


def render(input_file: Path = INPUT, output: Path = OUTPUT) -> Path:
    rows = read_csv(input_file)
    pm_stb = _aggregate(rows, "pm_stb")
    pm_css = _aggregate(rows, "pm_css")
    lc = _aggregate(rows, "lc_stb")

    use_style(scale=WIDE_TEXT_SCALE)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.7))
    figure.subplots_adjust(left=0.055, right=0.90, bottom=0.22, top=0.80, wspace=0.18)
    parameter_axis(axes[0], LABELS["linear_dependency"], empty_color=COLOR_PAPER_WHITE)
    parameter_axis(axes[1], LABELS["signatures"], empty_color=COLOR_PAPER_WHITE)
    parameter_axis(axes[2], "Local Clifford Equivalence", empty_color=COLOR_PAPER_WHITE)
    _draw(axes[0], (pm_stb, pm_css), "linear_dependency")
    _draw(axes[1], (pm_stb, pm_css), "signatures")
    _draw(axes[2], (lc,), "local_invariant")
    for ax, invariant in zip(axes[:2], ("linear_dependency", "signatures")):
        _annotate(
            ax,
            [
                f"Stabilizer: {_rate(_overall(pm_stb, invariant))} rejected",
                f"CSS: {_rate(_overall(pm_css, invariant))} rejected",
            ],
        )
    _annotate(axes[2], [f"Overall: {_rate(_overall(lc, 'local_invariant'))} rejected"])

    handles = [
        half_cell_key("left", COLOR_PAPER_GRAY_DARK, "Stabilizer codes"),
        half_cell_key("right", COLOR_PAPER_GRAY_DARK, "CSS codes"),
        Patch(
            facecolor=COLOR_PAPER_WHITE,
            edgecolor=COLOR_PAPER_GRAY_VERY_DARK,
            label="0% rejected",
        ),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=9)
    figure.suptitle(
        "Invariants' Rejection Patterns and Rates of Inequivalent Codes", fontsize=12 * WIDE_TEXT_SCALE, y=0.96
    )

    bar = figure.colorbar(scalar_mappable("Greys", Normalize(0, 1)), ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("Deeper color means\nmore rejected instances")
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    return save_png(figure, output)


if __name__ == "__main__":
    render()
