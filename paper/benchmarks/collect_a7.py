"""Collect z3 decision counts of the SAT encodings on CSS codes with a pinned X/Z rank split (A7).

Experiment 1:
rank_sweep solves CSS pairs with X-check rank rx in {0, 1, 2, r/2, r-2, r-1, r} in the
check-matrix encoding and general stabilizer pairs in the tableau encoding as reference.

Experiment 2:
row_mixing solves balanced CSS pairs (rx ≈ rz) in the tableau encoding, once as the clean
block-diagonal tableau and once after random row operations that mix X and Z generators.

The measure is the solver's decision count; the recorded times only serve as bookkeeping.
Rows are appended to sat_css_weakness.csv and existing keys are skipped on restart.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import z3

from benchmarks.experiments.utils import (
    _random_permutation,
    _random_row_space_base_change,
    _random_tableau_row_space_base_change,
    random_css_code,
    random_stabilizer_code,
)
from paper.benchmarks.common import COLLECTED_DIR, MASTER_SEED, append_row, completed_keys
from src.algorithms.p_css.p_css_sat import _build_peq_css_sat_solver
from src.algorithms.p_stb.p_stab_sat import _build_peq_stab_sat_solver
from src.core.css_code import CSSCode
from src.core.pauli import StabilizerTableau
from src.core.stabilizer_code import StabilizerCode

OUTPUT = COLLECTED_DIR / "sat_css_weakness.csv"
K = 4
NUM_SAMPLES = 10
TIMEOUT_SECONDS = 300.0
SWEEP_N = (14, 16, 18)
ROW_MIXING_N = (14, 16, 18)
ROW_MIXING_K = 2
ROW_MIXING_STEPS_PER_ROW = 30
VERBOSE = True

EXPERIMENT1 = "rank_sweep"
EXPERIMENT2 = "row_mixing"
MEASUREMENT = "base"  # measurement and probe are constant; both stay in the key so the file format is stable

RAW_FIELDS = (
    "experiment",
    "condition",
    "sample",
    "seed",
    "n",
    "k",
    "r",
    "rx",
    "rz",
    "measurement",
    "probe",
    "result",
    "timed_out",
    "timeout_seconds",
    "build_seconds",
    "solve_seconds",
    "decisions",
    "conflicts",
    "propagations",
    "rlimit_count",
    "assertions",
    "row_operation_variables",
)
KEY_FIELDS = ("experiment", "condition", "rx", "sample", "seed", "n", "k", "measurement", "probe")


@dataclass(frozen=True)
class Condition:
    experiment: str
    name: str
    rx: int | str
    rz: int | str
    builder: Callable[[], z3.Solver]
    row_operation_variables: int


def _rx_values(r: int) -> tuple[int, ...]:
    # in the order of the symbolic labels the extractor assigns (0, 1, 2, r/2, r-2, r-1, r)
    return (0, 1, 2, r // 2, r - 2, r - 1, r)


def _row_key(row) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def _sample_seed(master_seed: int, experiment: str, n: int, sample: int, rx: int | None = None) -> int:
    cell = f"{n}|{sample}" if rx is None else f"{n}|{rx}|{sample}"
    payload = f"A7|{master_seed}|{experiment}|{cell}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _stabilizer_partner(left: StabilizerCode, permutation: Sequence[int], row_seed: int) -> StabilizerCode:
    matrix = _random_row_space_base_change(np.asarray(left.symplectic, dtype=np.int8), seed=row_seed)
    columns = list(permutation) + [qubit + left.n for qubit in permutation]
    return StabilizerCode(StabilizerTableau(matrix[:, columns]))


def _css_partner(left: CSSCode, permutation: Sequence[int], row_seed: int) -> CSSCode:
    hx = _random_row_space_base_change(left.Hx, seed=row_seed)[:, permutation]
    hz = _random_row_space_base_change(left.Hz, seed=row_seed + 1)[:, permutation]
    return CSSCode(hx, hz)


def _permutation(n: int, rng: np.random.Generator) -> tuple[int, ...]:
    return _random_permutation(n, seed=int(rng.integers(0, 2**31)))


def _css_condition(n: int, k: int, rx: int, seed: int) -> Condition:
    rng = np.random.default_rng(seed)
    permutation = _permutation(n, rng)
    rz = n - k - rx
    left = random_css_code(n, k, rx=rx, seed=int(rng.integers(0, 2**31)))
    right = _css_partner(left, permutation, int(rng.integers(0, 2**31)))
    return Condition(EXPERIMENT1, "css", rx, rz, lambda: _build_peq_css_sat_solver(left, right), rx * rx + rz * rz)


def _general_condition(n: int, k: int, seed: int) -> Condition:
    rng = np.random.default_rng(seed)
    permutation = _permutation(n, rng)
    left = random_stabilizer_code(n, k, seed=int(rng.integers(0, 2**31)))
    right = _stabilizer_partner(left, permutation, int(rng.integers(0, 2**31)))
    return Condition(EXPERIMENT1, "general", "", "", lambda: _build_peq_stab_sat_solver(left, right), (n - k) ** 2)


def _row_mixing_conditions(n: int, k: int, seed: int) -> list[Condition]:
    """Balanced CSS pair as clean block-diagonal tableaus and as fully row-mixed tableaus."""
    rng = np.random.default_rng(seed)
    r = n - k
    rx = r // 2
    permutation = _permutation(n, rng)
    left = random_css_code(n, k, rx=rx, seed=int(rng.integers(0, 2**31)))
    right = _css_partner(left, permutation, int(rng.integers(0, 2**31)))
    clean = (StabilizerCode(left.generators), StabilizerCode(right.generators))
    mixed = tuple(
        StabilizerCode(
            _random_tableau_row_space_base_change(
                code.generators, seed=int(rng.integers(0, 2**31)), steps=ROW_MIXING_STEPS_PER_ROW * r
            )
        )
        for code in (left, right)
    )
    return [
        Condition(EXPERIMENT2, "clean", rx, r - rx, lambda: _build_peq_stab_sat_solver(*clean), r * r),
        Condition(EXPERIMENT2, "mixed", rx, r - rx, lambda: _build_peq_stab_sat_solver(*mixed), r * r),
    ]


def _measure(condition: Condition, *, sample: int, seed: int, n: int, k: int) -> dict[str, Any]:
    build_start = perf_counter()
    solver = condition.builder()
    build_seconds = perf_counter() - build_start
    solver.set(timeout=max(1, round(TIMEOUT_SECONDS * 1_000)))
    solver.set(random_seed=seed % (2**31 - 1))
    solve_start = perf_counter()
    result = solver.check()
    solve_seconds = perf_counter() - solve_start
    z3_statistics = solver.statistics()
    statistics = {key: z3_statistics.get_key_value(key) for key in z3_statistics.keys()}
    timed_out = result == z3.unknown and solver.reason_unknown() == "timeout"
    return {
        "experiment": condition.experiment,
        "condition": condition.name,
        "sample": sample,
        "seed": seed,
        "n": n,
        "k": k,
        "r": n - k,
        "rx": condition.rx,
        "rz": condition.rz,
        "measurement": MEASUREMENT,
        "probe": "",
        "result": str(result),
        "timed_out": timed_out,
        "timeout_seconds": TIMEOUT_SECONDS,
        "build_seconds": f"{build_seconds:.9f}",
        "solve_seconds": f"{solve_seconds:.9f}",
        "decisions": int(statistics.get("decisions", 0)),
        "conflicts": int(statistics.get("conflicts", 0)),
        "propagations": int(statistics.get("propagations", 0)),
        "rlimit_count": int(statistics.get("rlimit count", 0)),
        "assertions": len(solver.assertions()),
        "row_operation_variables": condition.row_operation_variables,
    }


def _collect(condition: Condition, *, sample: int, seed: int, n: int, k: int, completed: set[tuple[str, ...]]) -> None:
    identity = {
        "experiment": condition.experiment,
        "condition": condition.name,
        "rx": condition.rx,
        "sample": sample,
        "seed": seed,
        "n": n,
        "k": k,
        "measurement": MEASUREMENT,
        "probe": "",
    }
    if _row_key(identity) in completed:
        return
    row = _measure(condition, sample=sample, seed=seed, n=n, k=k)
    append_row(OUTPUT, row, RAW_FIELDS)
    completed.add(_row_key(row))
    if VERBOSE:
        print(
            f"A7 {condition.experiment} [[{n},{k}]] rx={condition.rx if condition.rx != '' else '-'} sample={sample} "
            f"{condition.name}: {row['result']}, decisions={row['decisions']}",
            flush=True,
        )


def collect() -> None:
    completed = completed_keys(OUTPUT, KEY_FIELDS)
    for n in SWEEP_N:
        for sample in range(NUM_SAMPLES):
            seed = _sample_seed(MASTER_SEED, EXPERIMENT1, n, sample)
            _collect(_general_condition(n, K, seed), sample=sample, seed=seed, n=n, k=K, completed=completed)
        for rx in _rx_values(n - K):
            for sample in range(NUM_SAMPLES):
                seed = _sample_seed(MASTER_SEED, EXPERIMENT1, n, sample, rx)
                _collect(_css_condition(n, K, rx, seed), sample=sample, seed=seed, n=n, k=K, completed=completed)
    for n in ROW_MIXING_N:
        for sample in range(NUM_SAMPLES):
            seed = _sample_seed(MASTER_SEED, EXPERIMENT2, n, sample)
            for condition in _row_mixing_conditions(n, ROW_MIXING_K, seed):
                _collect(condition, sample=sample, seed=seed, n=n, k=ROW_MIXING_K, completed=completed)


if __name__ == "__main__":
    collect()
