"""Extraction: statistics helpers and the per-experiment aggregation into by_cell tables."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.experiments.statistics import CSV_FIELDS
from paper.benchmarks.collect_a8 import RAW_FIELDS as A8_RAW_FIELDS
from paper.experiments.common import (
    aggregate_statistics,
    combine_statistic_rows,
    problem_for_algorithm,
    read_statistics,
    write_csv,
)
from paper.experiments.extract_a1 import extract as extract_a1
from paper.experiments.extract_a2 import extract as extract_a2, pairwise_refinement
from paper.experiments.extract_a3 import extract as extract_a3
from paper.experiments.extract_a4 import extract as extract_a4
from paper.experiments.extract_a5 import extract as extract_a5, select_winners
from paper.experiments.extract_a6 import extract as extract_a6
from paper.experiments.extract_a8 import extract as extract_a8

REJECTION_FIELDS = ("problem", "instance_id", "n", "k", "r", "invariant", "rejected", "status")


def _stat_row(algorithm: str, positive: bool, mean: float = 1.0, *, n: int = 3, k: int = 1, nr_seeds: int = 10,
              successful: int | None = None, timeouts: int = 0, memory_limited: int = 0,
              errors: int = 0, generation_errors: int = 0) -> dict[str, object]:
    return {
        "algorithm": algorithm, "generator": "fixture", "name": "", "n": n, "k": k, "positive": positive,
        "density": "", "symmetry": "", "seed": 42, "nr_seeds": nr_seeds, "timeout_seconds": 5400,
        "memory_limit_bytes": 1, "mean_seconds": mean, "stddev_seconds": 0, "maximum_seconds": mean,
        "num_cases": nr_seeds, "num_successful": nr_seeds if successful is None else successful,
        "num_unexpected": 0, "num_timeouts": timeouts, "num_memory_limited": memory_limited,
        "num_errors": errors, "num_generation_errors": generation_errors,
    }


def _algorithm(directory: Path, name: str, positive_mean: float, negative_mean: float, **kwargs) -> None:
    write_csv(directory / f"{name}.csv",
              [_stat_row(name, True, positive_mean, **kwargs), _stat_row(name, False, negative_mean, **kwargs)],
              CSV_FIELDS)


def _rejection_rows(n: int, k: int, rejected: dict[str, bool]) -> list[dict[str, object]]:
    return [{"problem": "pm_stb", "instance_id": "one", "n": n, "k": k, "r": n - k,
             "invariant": invariant, "rejected": flag, "status": "success"}
            for invariant, flag in rejected.items()]


def _invariant_timing_rows(problem: str, invariant: str, runtime: float) -> list[dict[str, object]]:
    return [{"problem": problem, "invariant": invariant, "seed": seed, "n": 3, "k": 1, "positive": positive,
             "runtime_seconds": runtime, "status": "success"}
            for positive in (True, False) for seed in range(5)]


def _method(algorithm: str, mean: float, *, complete: bool, timeouts: int = 0, errors: int = 0) -> dict[str, object]:
    return {
        "algorithm": algorithm, "problem": "pm_stb", "n": 5, "k": 2, "mean_seconds": mean, "complete": complete,
        "has_positive": True, "has_negative": True, "num_requested": 2, "num_successful": 2 - timeouts - errors,
        "num_timeouts": timeouts, "num_unexpected": 0, "num_memory_limited": 0, "num_errors": errors,
        "num_generation_errors": 0,
    }


# Statistics helpers ------------------------------------------------------------------------------

def test_read_statistics_keeps_latest_same_sized_invocation(tmp_path: Path) -> None:
    path = tmp_path / "statistics.csv"
    write_csv(path, [_stat_row("pm_stb_sat", True, 1.0, nr_seeds=3), _stat_row("pm_stb_sat", True, 2.0, nr_seeds=3)], CSV_FIELDS)

    rows = read_statistics(path)

    assert len(rows) == 1
    assert rows[0]["mean_seconds"] == "2.0"


def test_changed_seed_count_supersedes(tmp_path: Path) -> None:
    path = tmp_path / "statistics.csv"
    write_csv(path, [_stat_row("pm_stb_sat", positive, nr_seeds=nr_seeds)
                     for nr_seeds in (3, 5) for positive in (True, False)], CSV_FIELDS)

    aggregated = aggregate_statistics(read_statistics(path))

    assert len(aggregated) == 1
    assert aggregated[0]["num_requested"] == 10
    assert aggregated[0]["num_cases"] == 10


def test_combine_statistic_rows_pools_positive_and_negative() -> None:
    combined = combine_statistic_rows([
        {key: str(value) for key, value in _stat_row("pm_stb_sat", True, 1.0, nr_seeds=2).items()},
        {key: str(value) for key, value in _stat_row("pm_stb_sat", False, 3.0, nr_seeds=2).items()},
    ])

    assert combined["num_requested"] == 4
    assert combined["mean_seconds"] == pytest.approx(2.0)
    assert combined["complete"] is True


def test_problem_for_algorithm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="cannot infer problem family"):
        problem_for_algorithm("unknown_backend")


# A1 rejections -----------------------------------------------------------------------------------

def test_a1_computes_component_and_combined_rejections(tmp_path: Path) -> None:
    source = tmp_path / "rejections.csv"
    write_csv(source, _rejection_rows(3, 1, {"linear_dependency": False, "signatures": True}), REJECTION_FIELDS)

    cells = extract_a1(source, tmp_path / "a1")

    combined = next(row for row in cells if row["invariant"] == "combined")
    assert combined["num_rejected"] == 1
    assert combined["rejection_percentage"] == 100


def test_a1_keeps_measured_zero_rejections(tmp_path: Path) -> None:
    source = tmp_path / "rejections.csv"
    write_csv(source, _rejection_rows(3, 0, {"linear_dependency": False, "signatures": False}), REJECTION_FIELDS)

    cells = extract_a1(source, tmp_path / "a1")

    measured = [row for row in cells if row["invariant"] != "combined"]
    assert len(measured) == 2
    assert all(row["num_valid"] == 1 and row["num_rejected"] == 0 for row in measured)
    assert not any(row["n"] == 4 for row in cells)


# A2 signatures -----------------------------------------------------------------------------------

def test_a2_aggregates_random_code_instances(tmp_path: Path) -> None:
    source = tmp_path / "signatures.csv"
    write_csv(source, [
        {"problem": "pm_css", "seed": 89, "n": 3, "k": 1, "q_pairs": 1.0, "status": "success"},
        {"problem": "pm_css", "seed": 773, "n": 3, "k": 1, "q_pairs": 0.5, "status": "success"},
    ], ("problem", "seed", "n", "k", "q_pairs", "status"))

    cells = extract_a2(source, tmp_path / "a2.csv")

    assert len(cells) == 1
    assert cells[0]["num_valid"] == 2
    # For n=3, q=1 maps to 0 and q=1/2 maps to 3/4 after removing
    # unavoidable self-pairs and complementing; the two codes are averaged.
    assert cells[0]["mean_pairwise_refinement"] == pytest.approx(0.375)


def test_a2_pairwise_refinement_boundaries_and_intermediate_partition() -> None:
    assert pairwise_refinement(1.0, 7) == pytest.approx(0.0)
    assert pairwise_refinement(1 / 7, 7) == pytest.approx(1.0)
    # Class sizes 4 and 3 distinguish 2*4*3 of the 7*6 ordered distinct pairs.
    assert pairwise_refinement((4**2 + 3**2) / 7**2, 7) == pytest.approx(4 / 7)


# A3 relative invariant cost ----------------------------------------------------------------------

def test_a3_backend_choice_is_parameter_dependent(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_stb_sat", 2.0, 2.0)
    _algorithm(algorithms, "pm_stb_graph_iso", 1.0, 1.0)
    invariant_file = tmp_path / "invariants.csv"
    invariant_rows = _invariant_timing_rows("pm_stb", "signatures", 0.2)
    write_csv(invariant_file, invariant_rows, tuple(invariant_rows[0]))

    rows = extract_a3(invariant_file, algorithms, tmp_path / "a3.csv", ("pm_stb_sat", "pm_stb_graph_iso"))

    assert rows[0]["backend_algorithm"] == "pm_stb_graph_iso"
    assert rows[0]["relative_runtime"] == pytest.approx(0.2)


def test_a3_excludes_incomplete_invariant_cells(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "lc_stb_sat", 1.0, 1.0)
    invariant_file = tmp_path / "invariants.csv"
    invariant_rows = _invariant_timing_rows("lc_stb", "local_invariant", 0.2)
    invariant_rows[-1].update(status="generation_error", runtime_seconds="")
    write_csv(invariant_file, invariant_rows, tuple(invariant_rows[0]))

    assert extract_a3(invariant_file, algorithms, tmp_path / "a3.csv", ("lc_stb_sat",)) == []


# A4 graph representations ------------------------------------------------------------------------

def test_a4_reads_graph_representation_files_and_aggregates_polarities(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_stb_graph_iso", 1.0, 3.0)
    _algorithm(algorithms, "pm_css_matroid", 1.5, 3.5)
    _algorithm(algorithms, "lc_stb_graph_iso", 2.0, 4.0)
    _algorithm(algorithms, "pm_stb_sat", 0.01, 0.01)
    output = tmp_path / "a4.csv"

    rows = extract_a4(algorithms, output)

    assert {row["algorithm"] for row in rows} == {"pm_stb_graph_iso", "pm_css_matroid", "lc_stb_graph_iso"}
    assert next(row for row in rows if row["algorithm"] == "pm_stb_graph_iso")["mean_seconds"] == pytest.approx(2.0)
    assert output.is_file()


# A5 winners ----------------------------------------------------------------------------------------

def test_a5_selects_fastest_complete_algorithm_and_rejects_failed_one(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_stb_graph_iso", 2.0, 2.0)
    _algorithm(algorithms, "pm_stb_sat", 1.0, 1.0)
    _algorithm(algorithms, "pm_stb_bruteforce", 0.1, 0.1, successful=9, timeouts=1)

    winners = extract_a5(algorithms, tmp_path / "winners", ("pm_stb_graph_iso", "pm_stb_sat", "pm_stb_bruteforce"))

    assert winners[0]["winner"] == "pm_stb_sat"
    assert winners[0]["runner_up"] == "pm_stb_graph_iso"
    assert winners[0]["num_eligible_algorithms"] == 2
    assert winners[0]["selection"] == "completed"


def test_a5_uses_sat_when_only_timeout_and_memory_failed_methods_remain(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_css_sat", 100.0, 100.0, successful=9, timeouts=1)
    _algorithm(algorithms, "pm_css_matroid", 10.0, 10.0, successful=9, memory_limited=1)

    winners = extract_a5(algorithms, tmp_path / "winners", ("pm_css_sat", "pm_css_matroid"))

    assert winners[0]["winner"] == "pm_css_sat"
    assert winners[0]["selection"] == "timeout_fallback"
    assert winners[0]["winner_num_timeouts"] == 2


def test_a5_ignores_algorithm_outside_explicit_selection(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_css_sat", 100.0, 100.0, successful=9, timeouts=1)
    _algorithm(algorithms, "pm_css_graph_iso", 10.0, 10.0, successful=9, timeouts=1)

    winners = extract_a5(algorithms, tmp_path / "winners", ("pm_css_sat",))

    assert winners[0]["winner"] == "pm_css_sat"
    assert winners[0]["runner_up"] == ""
    assert winners[0]["selection"] == "timeout_fallback"


def test_a5_prefers_completed_and_excludes_errors() -> None:
    winners = select_winners([
        _method("pm_stb_sat", 2.0, complete=True),
        _method("pm_stb_bruteforce", 1.0, complete=False, timeouts=1),
        _method("pm_stb_graph_iso", 0.5, complete=False, errors=1),
    ])

    assert winners[0]["winner"] == "pm_stb_sat"
    assert winners[0]["selection"] == "completed"
    assert "pm_stb_graph_iso (errors)" in winners[0]["excluded_algorithms"]


def test_a5_missing_file_degrades_gracefully(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    algorithms = tmp_path / "algorithms"
    output = tmp_path / "a5"
    _algorithm(algorithms, "pm_stb_sat", 1.0, 1.0)

    winners = extract_a5(algorithms, output, algorithm_names=("pm_stb_sat", "pm_stb_aut"))

    assert winners[0]["winner"] == "pm_stb_sat"
    assert "pm_stb_aut (missing data)" in winners[0]["excluded_algorithms"]
    assert "excluding algorithms with missing collected data: pm_stb_aut" in capsys.readouterr().err
    assert (output / "by_cell.csv").stat().st_size > 0


# A6 CSS structure ---------------------------------------------------------------------------------

def test_a6_uses_two_complete_files_and_only_the_extra_css_file(tmp_path: Path) -> None:
    algorithms = tmp_path / "algorithms"
    _algorithm(algorithms, "pm_stb_sat", 4.0, 4.0)
    _algorithm(algorithms, "pm_css_sat", 1.0, 1.0)
    extra = tmp_path / "extra.csv"
    write_csv(extra, [_stat_row("pm_stb_sat_on_css", True, 10.0), _stat_row("pm_stb_sat_on_css", False, 10.0)], CSV_FIELDS)

    rows = extract_a6(algorithms, extra, tmp_path / "a6.csv")

    assert {row["variant"] for row in rows} == {"pm_stb_sat_on_stabilizer", "pm_css_sat_on_css", "pm_stb_sat_on_css"}
    assert all(row["num_requested"] == 20 for row in rows)
    comparison = next(row for row in rows if row["variant"] == "pm_stb_sat_on_css")
    assert comparison["hx_hz_log_scale_improvement_percentage"] > 0


# A8 hybrids ---------------------------------------------------------------------------------------

def _a8_raw_row(problem: str, code: str, positive: bool, seed: int, *, status: str = "success",
                runtime: float | str = 0.5, decided_by: str = "GI", stuck_at: str = "") -> dict[str, object]:
    return {"problem": problem, "code": code, "positive": positive, "seed": seed, "n": 2, "k": 0, "status": status,
            "runtime_seconds": runtime, "decided_by": decided_by, "stuck_at": stuck_at, "timeout_seconds": 60.0,
            "error": ""}


def test_a8_aggregates_runtimes_and_deciding_stages_per_cell(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_csv(tmp_path / "pm_stb_raw.csv", [
        _a8_raw_row("pm_stb", "bell", True, 89, runtime=1.0, decided_by="CI"),
        _a8_raw_row("pm_stb", "bell", True, 7, runtime=3.0, decided_by="GI"),
        _a8_raw_row("pm_stb", "bell", True, 5, runtime=2.0, decided_by="GI"),
        _a8_raw_row("pm_stb", "bell", False, 89, status="timeout", runtime=60.0, decided_by="", stuck_at="SAT"),
        _a8_raw_row("pm_stb", "bell", False, 7, status="error", runtime="", decided_by="", stuck_at="MI"),
    ], A8_RAW_FIELDS)

    cells = extract_a8(tmp_path, tmp_path / "by_cell.csv")

    assert [(cell["problem"], cell["code"], cell["positive"]) for cell in cells] == [("pm_stb", "bell", True), ("pm_stb", "bell", False)]
    positive, negative = cells
    assert (positive["num_cases"], positive["num_successful"], positive["r"]) == (3, 3, 2)
    assert positive["mean_seconds"] == pytest.approx(2.0) and positive["maximum_seconds"] == 3.0
    assert (positive["primary_decider"], positive["primary_decider_count"]) == ("GI", 2)
    assert (positive["secondary_decider"], positive["secondary_decider_count"]) == ("CI", 1)
    assert positive["deciders"] == "GI:2;CI:1"
    assert (negative["num_timeouts"], negative["num_errors"]) == (1, 1)
    assert negative["mean_seconds"] == 60.0  # timeouts enter at budget, errors are excluded
    assert negative["stuck_at"] == "MI:1;SAT:1"
    assert "pm_css_raw.csv is missing" in capsys.readouterr().err
    assert (tmp_path / "by_cell.csv").stat().st_size > 0


def test_a8_raises_without_any_raw_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_a8(tmp_path, tmp_path / "by_cell.csv")
