"""Collect z3 decision counts for the SAT encodings on CSS structure (A7).

Experiment 1 builds positive pairs in four conditions: unrestricted general
codes (A), general codes whose row transformation has two hidden blocks (B1),
the same pairs with both blocks exposed to the encoding (B2), and balanced CSS
codes with independent Rx and Rz (C). Besides the plain solve, each condition
is solved again with a few qubit mappings forced off the known witness
permutation, which measures how long a wrong mapping survives before UNSAT.
Experiment 2 solves clean CSS tableaus and fully row-mixed presentations of the
same stabilizer groups, both with the tableau encoding.

Rows are appended to a7_sat_css_structure.csv and existing keys are skipped on
restart.
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

OUTPUT = COLLECTED_DIR / "a7_sat_css_structure.csv"
K = 2
NUM_SAMPLES = 10
NUM_PROBES = 3
TIMEOUT_SECONDS = 300.0
EXPERIMENT1_N = (16, 18, 20)
EXPERIMENT2_N = (14, 16, 18)
VERBOSE = True

EXPERIMENT1 = "permutation_survival"
EXPERIMENT2 = "row_mixing"

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
    "source",
    "target",
    "witness_target",
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
KEY_FIELDS = (
    "experiment",
    "condition",
    "sample",
    "seed",
    "n",
    "k",
    "measurement",
    "probe",
)


@dataclass(frozen=True)
class Condition:
    experiment: str
    name: str
    builder: Callable[[], z3.Solver]
    witness: tuple[int, ...]
    row_operation_variables: int
    probe_mappings: bool


def _row_key(row) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def _sample_seed(master_seed: int, experiment: str, n: int, sample: int) -> int:
    payload = f"A7|{master_seed}|{experiment}|{n}|{sample}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _inverse_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for target, source in enumerate(permutation):
        inverse[source] = target
    return tuple(inverse)


def _stabilizer_partner(
    left: StabilizerCode,
    permutation: Sequence[int],
    row_seed: int,
    block_sizes: tuple[int, int] | None,
) -> StabilizerCode:
    matrix = np.asarray(left.symplectic, dtype=np.int8).copy()
    if block_sizes is None:
        matrix = _random_row_space_base_change(matrix, seed=row_seed)
    else:
        first, second = block_sizes
        matrix[:first] = _random_row_space_base_change(matrix[:first], seed=row_seed)
        matrix[first:] = _random_row_space_base_change(
            matrix[first:], seed=row_seed + 1
        )
    columns = list(permutation) + [qubit + left.n for qubit in permutation]
    return StabilizerCode(StabilizerTableau(matrix[:, columns]))


def _css_partner(
    left: CSSCode, permutation: Sequence[int], row_seed: int
) -> CSSCode:
    hx = _random_row_space_base_change(left.Hx, seed=row_seed)[:, permutation]
    hz = _random_row_space_base_change(left.Hz, seed=row_seed + 1)[:, permutation]
    return CSSCode(hx, hz)


def _exactly_one(variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    return z3.PbEq([(variable, 1) for variable in variables], 1)


def _xor(variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    value = z3.BoolVal(False)
    for variable in variables:
        value = z3.Xor(value, variable)
    return value


def _column_value(
    column: np.ndarray, variables: Sequence[z3.BoolRef]
) -> z3.BoolRef:
    return z3.And(
        *[
            variable if bit else z3.Not(variable)
            for bit, variable in zip(column, variables, strict=True)
        ]
    )


def _build_block_general_solver(
    left: StabilizerCode,
    right: StabilizerCode,
    block_sizes: tuple[int, int],
) -> z3.Solver:
    """Encode a general tableau with two explicit row-operation matrices."""
    solver = z3.Solver()
    n = left.n
    first, second = block_sizes
    r = first + second
    auxiliary = [
        z3.Bool(f"aux_{row}_{column}")
        for row in range(r)
        for column in range(2 * n)
    ]
    permutation = [
        z3.Bool(f"p_{source}_{target}")
        for source in range(n)
        for target in range(n)
    ]
    for source in range(n):
        solver.add(
            _exactly_one(
                [permutation[source * n + target] for target in range(n)]
            )
        )
    for target in range(n):
        solver.add(
            _exactly_one(
                [permutation[source * n + target] for source in range(n)]
            )
        )

    for source in range(n):
        for target in range(n):
            x_variables = [
                auxiliary[row * (2 * n) + target] for row in range(r)
            ]
            z_variables = [
                auxiliary[row * (2 * n) + target + n] for row in range(r)
            ]
            solver.add(
                z3.Implies(
                    permutation[source * n + target],
                    z3.And(
                        _column_value(left.symplectic[:, source], x_variables),
                        _column_value(left.symplectic[:, source + n], z_variables),
                    ),
                )
            )

    for block, (offset, size) in enumerate(((0, first), (first, second)), 1):
        coefficients = [
            z3.Bool(f"r{block}_{row}_{column}")
            for row in range(size)
            for column in range(size)
        ]
        for local_row in range(size):
            output_row = offset + local_row
            for column in range(2 * n):
                contributions = [
                    coefficients[local_row * size + contribution]
                    for contribution in range(size)
                    if right.symplectic[offset + contribution, column]
                ]
                solver.add(
                    auxiliary[output_row * (2 * n) + column]
                    == _xor(contributions)
                )
    return solver


def _experiment1_conditions(n: int, k: int, seed: int) -> list[Condition]:
    rng = np.random.default_rng(seed)
    r = n - k
    first, second = r // 2, r - r // 2
    permutation = tuple(
        int(value)
        for value in _random_permutation(n, seed=int(rng.integers(0, 2**31)))
    )
    witness = _inverse_permutation(permutation)

    general_left = random_stabilizer_code(
        n, k, seed=int(rng.integers(0, 2**31))
    )
    full_right = _stabilizer_partner(
        general_left, permutation, int(rng.integers(0, 2**31)), None
    )
    block_right = _stabilizer_partner(
        general_left,
        permutation,
        int(rng.integers(0, 2**31)),
        (first, second),
    )

    css_left = random_css_code(
        n, k, rx=first, seed=int(rng.integers(0, 2**31))
    )
    css_right = _css_partner(
        css_left, permutation, int(rng.integers(0, 2**31))
    )

    return [
        Condition(
            EXPERIMENT1,
            "A",
            lambda: _build_peq_stab_sat_solver(general_left, full_right),
            witness,
            r * r,
            True,
        ),
        Condition(
            EXPERIMENT1,
            "B1",
            lambda: _build_peq_stab_sat_solver(general_left, block_right),
            witness,
            r * r,
            True,
        ),
        Condition(
            EXPERIMENT1,
            "B2",
            lambda: _build_block_general_solver(
                general_left, block_right, (first, second)
            ),
            witness,
            first * first + second * second,
            True,
        ),
        Condition(
            EXPERIMENT1,
            "C",
            lambda: _build_peq_css_sat_solver(css_left, css_right),
            witness,
            first * first + second * second,
            True,
        ),
    ]


def _experiment2_conditions(n: int, k: int, seed: int) -> list[Condition]:
    rng = np.random.default_rng(seed)
    r = n - k
    first = r // 2
    permutation = tuple(
        int(value)
        for value in _random_permutation(n, seed=int(rng.integers(0, 2**31)))
    )
    witness = _inverse_permutation(permutation)
    css_left = random_css_code(
        n, k, rx=first, seed=int(rng.integers(0, 2**31))
    )
    css_right = _css_partner(
        css_left, permutation, int(rng.integers(0, 2**31))
    )
    clean_left = StabilizerCode(css_left.generators)
    clean_right = StabilizerCode(css_right.generators)
    mixed_left = StabilizerCode(
        _random_tableau_row_space_base_change(
            css_left.generators,
            seed=int(rng.integers(0, 2**31)),
            steps=30 * r,
        )
    )
    mixed_right = StabilizerCode(
        _random_tableau_row_space_base_change(
            css_right.generators,
            seed=int(rng.integers(0, 2**31)),
            steps=30 * r,
        )
    )
    return [
        Condition(
            EXPERIMENT2,
            "clean",
            lambda: _build_peq_stab_sat_solver(clean_left, clean_right),
            witness,
            r * r,
            False,
        ),
        Condition(
            EXPERIMENT2,
            "mixed",
            lambda: _build_peq_stab_sat_solver(mixed_left, mixed_right),
            witness,
            r * r,
            False,
        ),
    ]


def _measure(
    condition: Condition,
    *,
    sample: int,
    seed: int,
    n: int,
    k: int,
    timeout_seconds: float,
    measurement: str,
    probe: int | str = "",
    source: int | str = "",
    target: int | str = "",
) -> dict[str, Any]:
    build_start = perf_counter()
    solver = condition.builder()
    build_seconds = perf_counter() - build_start
    if source != "" and target != "":
        solver.add(z3.Bool(f"p_{source}_{target}"))
    solver.set(timeout=max(1, round(timeout_seconds * 1_000)))
    solver.set(random_seed=(seed + int(probe or 0)) % (2**31 - 1))
    solve_start = perf_counter()
    result = solver.check()
    solve_seconds = perf_counter() - solve_start
    z3_statistics = solver.statistics()
    statistics = {key: z3_statistics.get_key_value(key) for key in z3_statistics.keys()}
    timed_out = result == z3.unknown and solver.reason_unknown() == "timeout"
    r = n - k
    rx = r // 2
    return {
        "experiment": condition.experiment,
        "condition": condition.name,
        "sample": sample,
        "seed": seed,
        "n": n,
        "k": k,
        "r": r,
        "rx": rx,
        "rz": r - rx,
        "measurement": measurement,
        "probe": probe,
        "source": source,
        "target": target,
        "witness_target": condition.witness[int(source)] if source != "" else "",
        "result": str(result),
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "build_seconds": f"{build_seconds:.9f}",
        "solve_seconds": f"{solve_seconds:.9f}",
        "decisions": int(statistics.get("decisions", 0)),
        "conflicts": int(statistics.get("conflicts", 0)),
        "propagations": int(statistics.get("propagations", 0)),
        "rlimit_count": int(statistics.get("rlimit count", 0)),
        "assertions": len(solver.assertions()),
        "row_operation_variables": condition.row_operation_variables,
    }


def _probe_assignments(
    n: int, witness: Sequence[int], probes: int, seed: int
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed + 1)
    sources = rng.choice(n, size=min(probes, n), replace=False)
    alternatives = [int(rng.integers(0, n - 1)) for _ in sources]
    result = []
    for source_value, alternative in zip(sources, alternatives, strict=True):
        source = int(source_value)
        witness_target = witness[source]
        target = alternative + (alternative >= witness_target)
        result.append((source, target))
    return result


def _collect_conditions(
    conditions: Sequence[Condition],
    *,
    sample: int,
    seed: int,
    n: int,
    k: int,
    probes: int,
    timeout_seconds: float,
    output,
    completed: set[tuple[str, ...]],
    verbose: bool,
) -> None:
    assignments = _probe_assignments(n, conditions[0].witness, probes, seed)
    for condition in conditions:
        specifications: list[tuple[str, int | str, int | str, int | str]] = [
            ("base", "", "", "")
        ]
        if condition.probe_mappings:
            specifications.extend(
                ("invalid_mapping", probe, source, target)
                for probe, (source, target) in enumerate(assignments)
            )
        for measurement, probe, source, target in specifications:
            identity = {
                "experiment": condition.experiment,
                "condition": condition.name,
                "sample": sample,
                "seed": seed,
                "n": n,
                "k": k,
                "measurement": measurement,
                "probe": probe,
            }
            if _row_key(identity) in completed:
                continue
            row = _measure(
                condition,
                sample=sample,
                seed=seed,
                n=n,
                k=k,
                timeout_seconds=timeout_seconds,
                measurement=measurement,
                probe=probe,
                source=source,
                target=target,
            )
            append_row(output, row, RAW_FIELDS)
            completed.add(_row_key(row))
            if verbose:
                print(
                    f"A7 {condition.experiment} [[{n},{k}]] sample={sample} "
                    f"{condition.name} {measurement}: {row['result']}, "
                    f"decisions={row['decisions']}",
                    flush=True,
                )


def collect() -> None:
    completed = completed_keys(OUTPUT, KEY_FIELDS)
    experiments = (
        (EXPERIMENT1, EXPERIMENT1_N, _experiment1_conditions),
        (EXPERIMENT2, EXPERIMENT2_N, _experiment2_conditions),
    )
    for experiment, ns, conditions in experiments:
        for n in ns:
            for sample in range(NUM_SAMPLES):
                seed = _sample_seed(MASTER_SEED, experiment, n, sample)
                _collect_conditions(
                    conditions(n, K, seed), sample=sample, seed=seed, n=n, k=K,
                    probes=NUM_PROBES, timeout_seconds=TIMEOUT_SECONDS, output=OUTPUT,
                    completed=completed, verbose=VERBOSE,
                )


if __name__ == "__main__":
    collect()
