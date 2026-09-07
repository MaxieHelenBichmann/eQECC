"""Runtime and deciding-stage table of the hybrids (A8)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle

from paper.experiments.common import RESULTS_DIR, read_csv
from paper.experiments.extract_a8 import CODE_ORDER, PROBLEMS, STAGES
from paper.visualizations.common import (
    COLOR_PAPER_GRAY_DARK,
    COLOR_PAPER_GRAY_LIGHT,
    COLOR_PAPER_GRAY_VERY_LIGHT,
    COLOR_PAPER_GRAY_VERY_VERY_DARK,
    COLOR_PAPER_WHITE,
    RUNTIME_CMAP,
    TIMEOUT_SECONDS,
    decimal_ticks,
    mark_timeout,
    runtime_norm,
    save_png,
    scalar_mappable,
    use_style,
)

INPUT = RESULTS_DIR / "a8" / "by_cell.csv"
OUTPUT = RESULTS_DIR / "a8" / "a8.png"

PROBLEM_LABELS = {"pm_stb": "PM-STB", "pm_css": "PM-CSS", "lc_stb": "LC-STB"}
LABELS = ((True, "equivalent"), (False, "inequivalent"))
STAGE_LEGEND = (
    ("CI", "cheap invariants"),
    ("EI", "expensive invariants"),
    ("S", "signatures"),
    ("BF", "brute force"),
    ("MI", "matroid isomorphism"),
    ("GI", "graph isomorphism"),
    ("SAT", "SAT solver"),
    ("LSE", "graph-state LSE"),
)
# font sizes (stage label, count) of the first and second line in a cell
WINNER_SIZES = (7.4, 4.6)
RUNNER_UP_SIZES = (5.6, 4.0)


def _text_color(color) -> str:
    red, green, blue = to_rgb(color)
    return COLOR_PAPER_WHITE if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.46 else "#202020"


def _failures(row) -> int:
    return sum(
        int(row[field] or 0)
        for field in ("num_memory_limited", "num_errors", "num_unexpected", "num_generation_errors")
    )


def _stages(row) -> list[tuple[str, int]]:
    """Deciding and stuck-in stages together, most frequent first, ties in pipeline order."""
    counts: Counter[str] = Counter()
    for column in ("deciders", "stuck_at"):
        for entry in filter(None, row[column].split(";")):
            stage, count = entry.rsplit(":", 1)
            counts[stage] += int(count)
    order = {stage: index for index, stage in enumerate(STAGES)}
    return sorted(counts.items(), key=lambda item: (-item[1], order.get(item[0], len(order))))


def _stage_line(stage: str, count: int, total: int, sizes: tuple[float, float], color: str, *, bold: bool) -> HPacker:
    label = TextArea(
        stage, textprops={"fontsize": sizes[0], "color": color, "fontweight": "bold" if bold else "normal"}
    )
    share = TextArea(f"{count}/{total}", textprops={"fontsize": sizes[1], "color": color})
    return HPacker(children=[label, share], align="baseline", pad=0, sep=2.2)


def render(input_file: Path = INPUT, output_file: Path = OUTPUT) -> Path:
    rows = read_csv(input_file)
    cells = {(row["code"], row["problem"], row["positive"] == "True"): row for row in rows}
    labels = {row["code"]: row["code_label"] for row in rows}
    codes = sorted(labels, key=lambda code: (CODE_ORDER.get(code, len(CODE_ORDER)), code))
    runtimes = [float(row["mean_seconds"]) for row in rows if row["mean_seconds"]]
    timeout = max((float(row["timeout_seconds"]) for row in rows), default=TIMEOUT_SECONDS)
    norm = runtime_norm(runtimes, timeout=timeout)

    use_style()
    figure = plt.figure(figsize=(11.4, max(6.4, 0.39 * len(codes) + 1.9)))
    ax = figure.add_axes([0.035, 0.055, 0.78, 0.88])
    code_width, cell_width, header_rows = 1.7, 1.08, 2
    columns = [(problem, positive) for problem in PROBLEMS for positive, _ in LABELS]
    total_width = code_width + len(columns) * cell_width
    total_height = header_rows + len(codes)

    def box(x, y, width, height, color, **kwargs) -> None:
        ax.add_patch(
            Rectangle((x, y), width, height, facecolor=color, edgecolor=COLOR_PAPER_WHITE, linewidth=0.7, **kwargs)
        )

    box(0, 0, code_width, header_rows, COLOR_PAPER_GRAY_DARK)
    ax.text(0.10, 1.0, "Code", ha="left", va="center", fontsize=9, fontweight="bold")
    for index, problem in enumerate(PROBLEMS):
        x = code_width + index * len(LABELS) * cell_width
        box(x, 0, len(LABELS) * cell_width, 1, COLOR_PAPER_GRAY_DARK)
        ax.text(x + cell_width, 0.5, PROBLEM_LABELS[problem], ha="center", va="center", fontsize=8.5, fontweight="bold")
        for offset, (_, label) in enumerate(LABELS):
            box(x + offset * cell_width, 1, cell_width, 1, COLOR_PAPER_GRAY_LIGHT)
            ax.text(x + (offset + 0.5) * cell_width, 1.5, label, ha="center", va="center", fontsize=7.2)

    for row_index, code in enumerate(codes):
        y = header_rows + row_index
        box(0, y, code_width, 1, COLOR_PAPER_GRAY_VERY_LIGHT)
        ax.text(0.10, y + 0.5, labels[code], ha="left", va="center", fontsize=7.7)
        for column_index, (problem, positive) in enumerate(columns):
            x = code_width + column_index * cell_width
            cell = cells.get((code, problem, positive))
            runtime = float(cell["mean_seconds"]) if cell and cell["mean_seconds"] else None
            color = RUNTIME_CMAP(norm(runtime)) if runtime else COLOR_PAPER_GRAY_VERY_LIGHT
            failures = _failures(cell) if cell else 0
            box(x, y, cell_width, 1, color, hatch="///" if failures else None)
            text_color = _text_color(color)
            if cell is None:
                ax.text(
                    x + cell_width / 2,
                    y + 0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    fontweight="bold",
                    color=text_color,
                )
                continue
            stages, total = _stages(cell), int(cell["num_cases"])
            lines = [
                _stage_line(stage, count, total, sizes, text_color, bold=index == 0)
                for index, ((stage, count), sizes) in enumerate(zip(stages[:2], (WINNER_SIZES, RUNNER_UP_SIZES)))
            ]
            if not lines:
                lines = [_stage_line("\u2014", 0, total, WINNER_SIZES, text_color, bold=True)]
            ax.add_artist(
                AnnotationBbox(
                    VPacker(children=lines, align="center", pad=0, sep=1.6),
                    (x + cell_width / 2, y + 0.5),
                    frameon=False,
                    pad=0,
                    box_alignment=(0.5, 0.5),
                )
            )

    ax.add_patch(
        Rectangle(
            (0, 0),
            total_width,
            total_height,
            facecolor="none",
            edgecolor=COLOR_PAPER_GRAY_VERY_VERY_DARK,
            linewidth=0.7,
        )
    )
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis("off")
    figure.suptitle("Hybrid Runtime and Deciding Stage", x=0.43, y=0.98, fontsize=12)

    bar = figure.colorbar(scalar_mappable(RUNTIME_CMAP, norm), cax=figure.add_axes([0.855, 0.72, 0.021, 0.16]))
    decimal_ticks(bar)
    mark_timeout(bar, timeout)
    bar.set_label("Mean runtime [s]", labelpad=8)

    legend_ax = figure.add_axes([0.835, 0.10, 0.16, 0.55])
    legend_ax.axis("off")
    legend_ax.text(0, 1.0, "Deciding stage", ha="left", va="top", fontsize=9, fontweight="bold")
    lines = [f"{tag:<4} {label}" for tag, label in STAGE_LEGEND]
    lines += [
        "",
        "X a/b   stage X ended a of b runs",
        "        (decided or hit the budget)",
        "line 1  most frequent stage",
        "line 2  second most frequent",
    ]
    legend_ax.text(
        0,
        0.95,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=6.7,
        linespacing=1.35,
        family="monospace",
        color="#202020",
    )
    return save_png(figure, output_file)


if __name__ == "__main__":
    render()
