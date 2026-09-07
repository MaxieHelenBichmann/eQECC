"""Visualization: every paper figure renders from a synthetic by_cell table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from paper.experiments.common import write_csv
from paper.experiments.extract_a7 import EXPERIMENT1, EXPERIMENT2
from paper.experiments.extract_a8 import FIELDS as A8_FIELDS
from paper.visualizations import (
    visualize_a1, visualize_a2, visualize_a3, visualize_a4,
    visualize_a5, visualize_a6, visualize_a7, visualize_a8,
)

pytestmark = pytest.mark.filterwarnings("ignore:This figure includes Axes")


def _render(tmp_path: Path, rows: list[dict[str, object]], render, *extra_outputs: str) -> None:
    source = tmp_path / "by_cell.csv"
    output = tmp_path / "figure.png"
    write_csv(source, rows, tuple(rows[0]))
    assert render(source, output) == output
    assert output.stat().st_size > 0
    assert sorted(file.name for file in tmp_path.glob("*.png")) == sorted(("figure.png", *extra_outputs))


def test_a1_draws_measured_zero_with_an_outline() -> None:
    figure, ax = plt.subplots()
    visualize_a1._draw(ax, ({(3, 3, "linear_dependency"): (0, 1)},), "linear_dependency")
    try:
        assert len(ax.patches) == 2
        assert ax.patches[-1].get_facecolor()[-1] == 0
    finally:
        plt.close(figure)


def test_a1_rejections_write_main_and_overall_pngs(tmp_path: Path) -> None:
    _render(tmp_path, [
        {"problem": problem, "n": 3, "k": 1, "r": 2, "invariant": invariant, "num_valid": 20, "num_rejected": rejected}
        for problem, invariant, rejected in (
            ("pm_stb", "linear_dependency", 10), ("pm_css", "linear_dependency", 15),
            ("pm_stb", "signatures", 12), ("pm_css", "signatures", 18), ("lc_stb", "local_invariant", 8),
        )
    ], visualize_a1.render, "figure_overall.png")


def test_a2_signature_space(tmp_path: Path) -> None:
    _render(tmp_path, [
        {"problem": problem, "n": 3, "k": 1, "r": 2, "num_valid": 20, "num_censored": 0, "mean_pairwise_refinement": 0.5}
        for problem in ("pm_stb", "pm_css")
    ], visualize_a2.render)


def test_a3_relative_cost(tmp_path: Path) -> None:
    _render(tmp_path, [
        {
            "problem": problem, "invariant": invariant, "n": 3, "k": 1, "r": 2,
            "invariant_mean_seconds": 0.5, "invariant_stddev_seconds": 0.0,
            "backend_algorithm": f"{problem}_sat", "backend_mean_seconds": 1.0,
            "backend_selection": "timeout_fallback" if problem == "pm_css" else "completed",
            "backend_num_timeouts": 0, "relative_runtime": 0.5,
            "num_invariant_requested": 10, "num_invariant_successful": 10,
        }
        for problem, invariant in (
            ("pm_stb", "linear_dependency"), ("pm_css", "linear_dependency"),
            ("pm_stb", "signatures"), ("pm_css", "signatures"), ("lc_stb", "local_invariant"),
        )
    ], visualize_a3.render)


def test_a4_graph_representations(tmp_path: Path) -> None:
    _render(tmp_path, [
        {
            "algorithm": algorithm, "n": 3, "r": 2, "mean_seconds": 1.0, "num_successful": 1, "num_timeouts": 0,
            "num_memory_limited": 0, "num_errors": int(algorithm == "lc_stb_graph_iso"), "num_unexpected": 0,
        }
        for algorithm in ("pm_stb_graph_iso", "lc_stb_graph_iso", "pm_css_matroid")
    ], visualize_a4.render)


def test_a5_winner_maps(tmp_path: Path) -> None:
    _render(tmp_path, [
        {
            "problem": problem, "n": 3, "k": 1, "r": 2, "winner": f"{problem}_sat", "mean_seconds": 1.0,
            "runner_up": "", "runner_up_mean_seconds": "", "speed_ratio": "", "num_eligible_algorithms": 1,
            "selection": "timeout_fallback" if problem == "pm_css" else "completed",
            "excluded_algorithms": "pm_stb_aut (missing data)" if problem == "pm_stb" else "",
        }
        for problem in ("pm_stb", "pm_css", "lc_stb")
    ], visualize_a5.render)


def test_a6_css_structure(tmp_path: Path) -> None:
    _render(tmp_path, [
        {
            "variant": variant, "n": 3, "r": 2, "mean_seconds": 1.0,
            "hx_hz_log_scale_improvement_percentage": 10 if variant == "pm_stb_sat_on_css" else "",
            "num_successful": 1, "num_timeouts": 0, "num_memory_limited": 0, "num_errors": 0,
        }
        for variant in ("pm_stb_sat_on_stabilizer", "pm_css_sat_on_css", "pm_stb_sat_on_css")
    ], visualize_a6.render, "figure_css.png")


def test_a7_decision_counts(tmp_path: Path) -> None:
    _render(tmp_path, [
        {"experiment": experiment, "condition": condition, "n": 8, "r": 4,
         "median_base_decisions": 10, "median_invalid_mapping_decisions": 5}
        for experiment, conditions in ((EXPERIMENT1, ("A", "B1", "B2", "C")), (EXPERIMENT2, ("clean", "mixed")))
        for condition in conditions
    ], visualize_a7.render)


def test_a8_hybrid_stages(tmp_path: Path) -> None:
    def cell(problem: str, code: str, positive: bool, **overrides) -> dict[str, object]:
        defaults = dict(problem=problem, code=code, code_label=code.title(), n=2, k=0, r=2, positive=positive,
                        num_cases=2, num_successful=2, mean_seconds=0.5, stddev_seconds=0.0, maximum_seconds=0.5,
                        primary_decider="GI", primary_decider_count=2, secondary_decider="", secondary_decider_count=0,
                        deciders="GI:2", stuck_at="", timeout_seconds=60.0)
        return dict.fromkeys(A8_FIELDS, 0) | defaults | overrides

    _render(tmp_path, [
        cell("pm_stb", "bell", True),
        cell("pm_stb", "bell", False, num_successful=0, num_timeouts=1, num_errors=1, mean_seconds=60.0,
             maximum_seconds=60.0, primary_decider="", primary_decider_count=0, deciders="", stuck_at="SAT:1;MI:1"),
        cell("pm_css", "steane", True, n=7, k=1, r=6, primary_decider="MI", deciders="MI:2"),
        cell("lc_stb", "steane", False, n=7, k=1, r=6, num_successful=1, num_generation_errors=1,
             primary_decider="LSE", primary_decider_count=1, deciders="LSE:1"),
    ], visualize_a8.render)
