"""Median solver decisions against the X/Z rank split, and for clean vs row-mixed CSS tableaus (A7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from paper.experiments.common import RESULTS_DIR, as_float, as_int, read_csv
from paper.experiments.extract_a7 import OUTPUT as INPUT
from paper.experiments.extract_a7 import ROW_MIXING_OUTPUT, RX_LABELS
from paper.visualizations.common import (
    COLOR_PAPER_GRAY_DARK,
    COLOR_PAPER_GRAY_LIGHT,
    COLOR_PAPER_GRAY_VERY_DARK,
    COLOR_PAPER_GRAY_VERY_LIGHT,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    COLOR_PAPER_WHITE,
    save_png,
    use_style,
)

ROW_MIXING_INPUT = ROW_MIXING_OUTPUT
OUTPUT = RESULTS_DIR / "a7" / "a7.png"

COLORS = ("#D8A0DC", "#8F5FC2", "#2F1F52")
STYLES = (("css", "--", "s", "CSS codes"),)


def _render_sweep(ax, rows) -> None:
    ns = sorted({as_int(row["n"]) for row in rows})
    cells = {(as_int(row["n"]), row["condition"], row["rx_label"]): row for row in rows}
    positions = {label: index for index, label in enumerate(RX_LABELS)}
    for n, color in zip(ns, COLORS, strict=False):
        for condition, style, marker, _ in STYLES:
            points = [
                (positions[label], as_float(cells[(n, condition, label)]["median_decisions"]))
                for label in RX_LABELS
                if (n, condition, label) in cells and as_float(cells[(n, condition, label)]["median_decisions"])
            ]
            ax.plot(
                *zip(*points, strict=True), linestyle=style, marker=marker, markersize=5, color=color, linewidth=1.1
            )
        general = cells.get((n, "general", "general"))
        if general is not None and as_float(general["median_decisions"]) is not None:
            ax.axhline(as_float(general["median_decisions"]), linestyle="-", color=color, linewidth=1.0, zorder=2)

    ax.set_yscale("log")
    ax.set_xticks(range(len(RX_LABELS)), [label.replace("r/2", "⌊r/2⌋") for label in RX_LABELS])
    ax.set_xlabel("X-check rank $r_x$ (Z-check rank $r_z = r - r_x$)")
    ax.set_ylabel("Median solver decisions")
    ax.set_title("Rank sweep of positive CSS pairs, $k=4$", pad=6)
    ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=color, linewidth=1.1, label=f"$n={n}$") for n, color in zip(ns, COLORS)]
    handles += [
        Line2D([], [], color=COLOR_PAPER_GRAY_VERY_VERY_DARK, linestyle=style, marker=marker, markersize=4, label=label)
        for _, style, marker, label in STYLES
    ]
    handles.append(Line2D([], [], color=COLOR_PAPER_GRAY_VERY_VERY_DARK, linestyle="-", label="Non-CSS codes"))
    
    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        columnspacing=1.6,
        handlelength=2.0,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor=COLOR_PAPER_WHITE,
        edgecolor=COLOR_PAPER_GRAY_VERY_DARK,
    )


def _format_decisions(value) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}" if float(value).is_integer() else f"{float(value):,.1f}"


def _cell(ax, x, y, width, height, text, *, facecolor, weight="normal", fontsize=8.0) -> None:
    ax.add_patch(Rectangle((x, y), width, height, facecolor=facecolor, edgecolor=COLOR_PAPER_WHITE, linewidth=1.0))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontweight=weight, fontsize=fontsize)


def _render_row_mixing(ax, rows) -> None:
    """Clean block-diagonal vs fully row-mixed CSS tableaus, both in the tableau encoding, as a table."""
    cells = {(as_int(row["n"]), row["condition"]): row for row in rows}
    ns = sorted({n for n, _ in cells})
    k = as_int(rows[0]["k"]) if rows else 2
    widths = (0.14, 0.14, 0.32, 0.40)
    positions = [0.0]
    for width in widths:
        positions.append(positions[-1] + width)
    header_height, row_height = 0.8, 0.7
    total_height = header_height + len(ns) * row_height
    for index, header in enumerate(("n", "r", "clean", "mixed")):
        _cell(
            ax,
            positions[index],
            0,
            widths[index],
            header_height,
            header,
            facecolor=COLOR_PAPER_GRAY_DARK,
            weight="bold",
        )
    for index, n in enumerate(ns):
        y = header_height + index * row_height
        color = COLOR_PAPER_GRAY_VERY_LIGHT if index % 2 == 0 else COLOR_PAPER_GRAY_LIGHT
        clean, mixed = cells[(n, "clean")], cells[(n, "mixed")]
        values = (
            str(n),
            clean["r"],
            _format_decisions(as_float(clean["median_decisions"])),
            _format_decisions(as_float(mixed["median_decisions"])),
        )
        for column, value in enumerate(values):
            _cell(ax, positions[column], y, widths[column], row_height, value, facecolor=color)
    ax.add_patch(
        Rectangle((0, 0), 1, total_height, facecolor="none", edgecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK, linewidth=0.8)
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(total_height, 0)
    ax.axis("off")
    ax.set_title(f"Clean vs row-mixed CSS tableaus, $k={k}$", pad=6)


def render(input_file: Path = INPUT, row_mixing_file: Path = ROW_MIXING_INPUT, output: Path = OUTPUT) -> Path:
    use_style()
    figure = plt.figure(figsize=(10.4, 4.2))
    left = figure.add_axes([0.065, 0.13, 0.62, 0.75])
    right = figure.add_axes([0.715, 0.36, 0.275, 0.38])
    _render_sweep(left, read_csv(input_file))
    _render_row_mixing(right, read_csv(row_mixing_file))
    return save_png(figure, output)


if __name__ == "__main__":
    render()
