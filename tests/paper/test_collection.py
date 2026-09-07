"""Data collection: certified pair generation, the collectors and their CSV persistence."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from benchmarks.experiments.generators_structured import load_named_code
from benchmarks.experiments.run import RunResult
from paper.benchmarks import collect_a1, collect_a2, collect_a3, collect_a8, collect_algorithm, common
from paper.benchmarks.common import certified_negative_pair, css_certifier, invariant_matrices
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.css_code import CSSCode
from src.core.stabilizer_code import StabilizerCode


def _run_result(result) -> RunResult:
    return RunResult(
        runtime=0.1,
        result=result,
        expected=None,
        result_is_expected=False,
        timed_out=False,
        memory_exceeded=False,
        error=None,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# Suite and certified negatives -----------------------------------------------------------------


def test_fixed_suite_matches_the_thesis_grid_and_seed_schedule() -> None:
    assert len(collect_a1.DIMENSIONS) == 185
    assert collect_a1.DIMENSIONS[:3] == ((3, 0), (3, 1), (3, 2))
    assert collect_a1.SEEDS == (89, 773, 654, 438, 433, 858, 85, 697, 201, 94)


def test_only_active_public_collectors_remain() -> None:
    directory = Path(__file__).resolve().parents[2] / "paper" / "benchmarks"
    assert sorted(path.name for path in directory.glob("collect_*.py")) == [
        "collect_a1.py",
        "collect_a2.py",
        "collect_a3.py",
        "collect_a6.py",
        "collect_a7.py",
        "collect_a8.py",
        "collect_algorithm.py",
    ]


def test_clifford_perturbed_stabilizer_negative_is_certified() -> None:
    left, right = certified_negative_pair("pm_stb", 3, 1, 89, max_attempts=5)
    assert are_peq_stab_sat(left, right) is False


def test_independent_css_negative_has_a_complete_certificate() -> None:
    left, right = certified_negative_pair("pm_css", 3, 1, 89, max_attempts=5)
    assert isinstance(left, CSSCode) and isinstance(right, CSSCode)
    assert (left.Hx.shape[0], left.Hz.shape[0]) == (right.Hx.shape[0], right.Hz.shape[0])
    assert are_peq_css_sat(left, right) is False


def test_cnot_perturbed_css_negative_keeps_check_ranks() -> None:
    for seed in range(3):
        left, right = certified_negative_pair("pm_css", 7, 3, seed, css_cnots=True)
        assert isinstance(left, CSSCode) and isinstance(right, CSSCode)
        assert (left.Hx.shape[0], left.Hz.shape[0]) == (right.Hx.shape[0], right.Hz.shape[0])


def test_stabilizer_candidates_use_clifford_perturbations(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = (object(), object())
    calls = []

    def candidate(*args, **kwargs):
        calls.append((args, kwargs))
        return pair

    monkeypatch.setattr(common.NonPEqCodePairGenerator, "stabilizer_codes_clifford_candidate", candidate)
    monkeypatch.setattr(common, "certified_inequivalent", lambda *args: True)
    assert certified_negative_pair("pm_stb", 7, 3, 89) is pair
    assert certified_negative_pair("lc_stb", 7, 3, 89) is pair
    assert len(calls) == 2
    assert all(kwargs["gate_steps"] == common.GATE_STEPS for _, kwargs in calls)


def test_css_certifier_selection_respects_backend_limits() -> None:
    assert css_certifier(47, 38) is are_peq_css_sat  # r = 9
    assert css_certifier(28, 18) is common.are_peq_css_matroid
    assert css_certifier(29, 19) is None


def test_large_high_rank_css_uses_certified_generator_without_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = (object(), object())
    monkeypatch.setattr(common.NonPEqCodePairGenerator, "css_codes_cascaded", lambda *args: pair)
    monkeypatch.setattr(
        common, "certified_inequivalent", lambda *args: pytest.fail("large CSS fallback must not invoke a backend")
    )
    assert certified_negative_pair("pm_css", 29, 19, 89, max_attempts=1) is pair


# A1 rejections and A2 signatures -----------------------------------------------------------------


def test_raw_collector_persists_rows_and_resumes_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "rejections.csv"
    monkeypatch.setattr(collect_a1, "PROBLEMS", ("pm_stb",))
    monkeypatch.setattr(collect_a1, "certified_negative_pair", lambda *args: (object(), object()))
    monkeypatch.setattr(collect_a1, "run", lambda *args, **kwargs: _run_result(False))

    first = collect_a1.collect(dimensions=[(3, 1)], seeds=[89], output_file=output)
    second = collect_a1.collect(dimensions=[(3, 1)], seeds=[89], output_file=output)

    assert len(first) == 2
    assert second == []
    assert len(_rows(output)) == 2


def test_signature_collector_covers_both_families_and_full_grid() -> None:
    assert collect_a2.PROBLEMS == ("pm_stb", "pm_css")
    assert set(collect_a2.DIMENSIONS) == set(collect_a2.measurement_dimensions())
    ranks = [n - k for n, k in collect_a2.DIMENSIONS]
    assert ranks == sorted(ranks)


def test_signature_collector_generates_one_code_per_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "signatures.csv"
    generated = []

    def generate_random_code(*args):
        generated.append(args)
        return object(), 1

    assert not hasattr(collect_a2, "NonPEqCodePairGenerator")
    monkeypatch.setattr(collect_a2, "PROBLEMS", ("pm_css",))
    monkeypatch.setattr(collect_a2, "generate_random_code", generate_random_code)
    monkeypatch.setattr(collect_a2, "run", lambda *args, **kwargs: _run_result([2, 1]))

    first = collect_a2.collect(dimensions=[(3, 1)], seeds=[89], output_file=output)
    second = collect_a2.collect(dimensions=[(3, 1)], seeds=[89], output_file=output)

    persisted = _rows(output)
    assert generated == [("pm_css", 3, 1, 89)]
    assert len(first) == 1 and second == []
    assert len(persisted) == 1
    assert "positive" not in persisted[0]
    assert persisted[0]["x_rank"] == "1"


def test_signature_metric_has_the_expected_extremes() -> None:
    assert collect_a2.signature_metric([7], 7) == pytest.approx(1.0)
    assert collect_a2.signature_metric([1] * 7, 7) == pytest.approx(1 / 7)


# A3 invariant timings ----------------------------------------------------------------------------


def test_invariant_timing_generator_certifies_locally_before_preparing(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = (StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(3))
    prepared = (object(), object())
    events: list[str] = []

    def recording(event: str, result):
        def stub(*args):
            events.append(event)
            return result

        return stub

    monkeypatch.setattr(collect_a3, "certified_negative_pair", recording("certify", pair))
    monkeypatch.setattr(collect_a3, "invariant_matrices", recording("prepare", prepared))

    generated = collect_a3.generate_pair("lc_stb", 3, 0, False, 89)

    assert collect_a3.invariant_matrices("lc_stb", *generated) is prepared
    assert events == ["certify", "prepare"]


def test_negative_certification_failure_becomes_generation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def failing(*args):
        calls.append(args)
        raise RuntimeError("certification timed out")

    monkeypatch.setattr(collect_a3, "certified_negative_pair", failing)
    monkeypatch.setattr(collect_a3, "INVARIANTS", {"lc_stb": ("local_invariant",)})
    monkeypatch.setattr(collect_a3, "VERBOSE", False)

    rows = collect_a3.collect(dimensions=[(3, 0)], seeds=(89,), output_file=tmp_path / "timings.csv")

    negative = next(row for row in rows if row["positive"] is False)
    assert negative["status"] == "generation_error"
    assert negative["runtime_seconds"] is None
    assert calls == [("lc_stb", 3, 0, 89)]


def test_prepared_matrix_arity_matches_each_invariant_family() -> None:
    arity = {
        problem: len(invariant_matrices(problem, *collect_a3.generate_pair(problem, 3, 1, True, 89)))
        for problem in ("pm_stb", "pm_css", "lc_stb")
    }
    assert arity == {"pm_stb": 2, "pm_css": 4, "lc_stb": 2}


# Algorithm collector -----------------------------------------------------------------------------


def test_runtime_negative_uses_local_a1_style_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    pair = (object(), object())
    calls: list[tuple[object, ...]] = []

    def certified_negative_pair(*args):
        calls.append(args)
        return pair

    monkeypatch.setattr(collect_algorithm, "certified_negative_pair", certified_negative_pair)

    first = collect_algorithm.CertifiedRandomCaseGenerator("pm_stb_sat", 7, 3, False)(89)
    second = collect_algorithm.CertifiedRandomCaseGenerator("pm_stb_graph_iso", 7, 3, False)(89)

    assert first.inputs == second.inputs == pair
    assert calls == [("pm_stb", 7, 3, 89), ("pm_stb", 7, 3, 89)]


def test_algorithm_collector_cli_selects_algorithms() -> None:
    assert vars(collect_algorithm.parse_args(["--algorithm", "pm_stb_sat"])) == {"algorithm": ["pm_stb_sat"]}
    assert collect_algorithm.parse_args([]).algorithm == list(collect_algorithm.ALGORITHM_N_RANGES)


def test_algorithm_collector_chooses_output_file_from_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(collect_algorithm, "OUTPUT_DIRECTORY", tmp_path)
    monkeypatch.setattr(collect_algorithm, "VERBOSE", False)
    monkeypatch.setattr(collect_algorithm, "measurement_dimensions", lambda *args: [(3, 1)])
    monkeypatch.setattr(collect_algorithm, "run_statistics", lambda *args, **kwargs: calls.append((args, kwargs)))

    collect_algorithm.collect(["pm_stb_sat"])

    assert len(calls) == 2
    args, _ = calls[0]
    assert args[0] is collect_algorithm.ALGORITHMS["pm_stb_sat"]
    assert (args[1].n, args[1].k, args[1].positive) == (3, 1, True)
    assert args[2:4] == (collect_algorithm.MASTER_SEED, collect_algorithm.NUM_SEEDS)
    assert args[4] == tmp_path / "pm_stb_sat.csv"
    assert calls[1][0][1].positive is False


def test_automorphism_collection_skips_before_writing_without_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(collect_algorithm.shutil, "which", lambda _: None)
    monkeypatch.setattr(collect_algorithm, "OUTPUT_DIRECTORY", tmp_path)

    collect_algorithm.collect(("pm_stb_aut",))

    assert not list(tmp_path.iterdir())
    assert "skipping pm_stb_aut" in capsys.readouterr().out


# A8 hybrids on named codes -----------------------------------------------------------------------

A8_SMALL_CODES = ("bell", "3q_rep", "5q_prf")
A8_KWARGS = dict(generation_timeout=60.0, timeout=60.0, memory_limit_bytes=4 * 1024**3, verbose=False)


@pytest.mark.parametrize("code_name", ("steane", "5q_prf"))
def test_a8_code_serialization_round_trips(code_name: str) -> None:
    code = load_named_code(code_name)
    restored = collect_a8.decode_code(collect_a8.encode_code(code), code.n)
    assert isinstance(restored, CSSCode) == isinstance(code, CSSCode)
    assert (restored.n, restored.k) == (code.n, code.k)
    assert np.array_equal(restored.symplectic % 2, code.symplectic % 2)


@pytest.mark.parametrize("problem", tuple(collect_a8.HYBRIDS))
@pytest.mark.parametrize("positive", (True, False))
def test_a8_instances_are_deterministic_and_correctly_labeled(problem: str, positive: bool) -> None:
    for code_name in A8_SMALL_CODES:
        if problem == "pm_css" and code_name in collect_a8.NON_CSS_CODES:
            continue
        first = collect_a8.generate_instance(problem, code_name, positive, 89)
        assert first == collect_a8.generate_instance(problem, code_name, positive, 89)
        left = collect_a8.decode_code(first["left"], first["n"])
        right = collect_a8.decode_code(first["right"], first["n"])
        assert (left.n, left.k) == (right.n, right.k)
        assert collect_a8.certified_inequivalent(problem, left, right) is not positive


def test_a8_pm_hybrids_share_the_positive_css_instance() -> None:
    stb = collect_a8.generate_instance("pm_stb", "steane", True, 89)
    css = collect_a8.generate_instance("pm_css", "steane", True, 89)
    assert (stb["left"], stb["right"]) == (css["left"], css["right"])


def test_a8_read_trace_reports_last_stage_of_a_killed_call(tmp_path: Path) -> None:
    log = tmp_path / "trace.log"
    log.write_text("CI\nEI\nS\n", encoding="utf-8")
    assert collect_a8.read_trace(log) == (["CI", "EI", "S"], "")
    log.write_text("CI\nSAT\n#decided_by SAT\n", encoding="utf-8")
    assert collect_a8.read_trace(log) == (["CI", "SAT"], "SAT")


def test_a8_collector_resumes_from_raw_and_instance_csvs(tmp_path: Path) -> None:
    kwargs = dict(codes=("bell", "5q_prf"), seeds=(89, 7), output_directory=tmp_path, **A8_KWARGS)
    collect_a8.collect(("pm_stb", "pm_css"), **kwargs)

    raw = _rows(tmp_path / "pm_stb_raw.csv")
    assert len(raw) == 8
    assert {row["status"] for row in raw} == {"success"}
    assert all(row["decided_by"] for row in raw)
    assert len(_rows(tmp_path / "pm_css_raw.csv")) == 4  # 5q_prf is not CSS
    instances = _rows(tmp_path / "pm_stb_instances.csv")
    assert len(instances) == 8

    # Re-running measures nothing new; deleting the raw file reuses the cached instances.
    collect_a8.collect(("pm_stb",), **kwargs)
    assert len(_rows(tmp_path / "pm_stb_raw.csv")) == 8
    (tmp_path / "pm_stb_raw.csv").unlink()
    collect_a8.collect(("pm_stb",), **kwargs)
    assert len(_rows(tmp_path / "pm_stb_raw.csv")) == 8
    assert _rows(tmp_path / "pm_stb_instances.csv") == instances


def test_a8_generation_failure_is_recorded_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(problem: str, code_name: str, positive: bool, seed: int) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(collect_a8, "generate_instance", broken)
    collect_a8.collect(("lc_stb",), codes=("bell",), seeds=(1,), output_directory=tmp_path, **A8_KWARGS)

    raw = _rows(tmp_path / "lc_stb_raw.csv")
    assert [row["status"] for row in raw] == ["generation_error", "generation_error"]
    assert "boom" in raw[0]["error"]
    assert [row["status"] for row in _rows(tmp_path / "lc_stb_instances.csv")] == ["generation_error"] * 2
