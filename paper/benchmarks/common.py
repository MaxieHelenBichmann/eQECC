"""Constants and helpers shared by the collectors."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.experiments.generators_random import NonPEqCodePairGenerator
from benchmarks.experiments.run import RunResult, run
from src.algorithms.lc_stb.lc_stb_sat import are_lceq_sat
from src.algorithms.p_css.p_css_matroid import are_peq_css_matroid
from src.algorithms.p_css.p_css_sat import are_peq_css_sat
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat
from src.core.stabilizer_code import StabilizerCode
from src.hybrids import lc_stb, p_css, p_stab

ROOT = Path(__file__).resolve().parents[2]
COLLECTED_DIR = ROOT / "paper" / "data" / "collected"

MASTER_SEED = 42
TIMEOUT_SECONDS = 5_400.0
CERTIFICATION_TIMEOUT_SECONDS = 600.0
MEMORY_LIMIT_BYTES = 13 * 1024**3
CSS_SAT_MAX_R = 9
CSS_MATROID_MAX_N = 28
GATE_STEPS = 2

INVARIANTS = {
    "pm_stb": ("linear_dependency", "signatures"),
    "pm_css": ("linear_dependency", "signatures"),
    "lc_stb": ("local_invariant",),
}
CERTIFIERS = {"pm_stb": are_peq_stab_sat, "lc_stb": are_lceq_sat}
CodePair = tuple[StabilizerCode, StabilizerCode]


def execution_status(result: RunResult) -> str:
    if result.timed_out:
        return "timeout"
    if result.memory_exceeded:
        return "memory_limited"
    if result.error is not None:
        return "error"
    return "success"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_keys(path: Path, fields: Sequence[str]) -> set[tuple[str, ...]]:
    return {tuple(row[field] for field in fields) for row in read_rows(path)}


def append_row(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def css_certifier(n: int, k: int):
    if n - k <= CSS_SAT_MAX_R:
        return are_peq_css_sat
    if n <= CSS_MATROID_MAX_N:
        return are_peq_css_matroid
    return None


def attempt_seed(problem: str, n: int, k: int, seed: int, attempt: int) -> int:
    value = f"{problem}_negative_matching=False|{n}|{k}|{seed}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def certified_inequivalent(problem: str, pair: CodePair, n: int, k: int) -> bool:
    certifier = css_certifier(n, k) if problem == "pm_css" else CERTIFIERS[problem]
    result = run(
        certifier, pair, False,
        timeout=CERTIFICATION_TIMEOUT_SECONDS, max_memory_bytes=MEMORY_LIMIT_BYTES,
    )
    if result.timed_out:
        raise RuntimeError("inequivalence certification timed out")
    if result.memory_exceeded:
        raise RuntimeError("inequivalence certification exceeded memory limit")
    if result.error is not None:
        raise RuntimeError(f"inequivalence certification failed: {result.error}")
    return result.result is False


def certified_negative_pair(
    problem: str, n: int, k: int, seed: int, *, css_cnots: bool = False, max_attempts: int = 1_000
) -> CodePair:
    """Random source code plus a perturbed copy that an exact backend proved inequivalent.

    CSS codes without a feasible exact certifier come from the cascaded generator,
    which carries its own certificate.
    """
    if problem == "pm_css" and css_certifier(n, k) is None:
        return NonPEqCodePairGenerator.css_codes_cascaded(n, k, attempt_seed(problem, n, k, seed, 0))
    for attempt in range(max_attempts):
        candidate_seed = attempt_seed(problem, n, k, seed, attempt)
        if problem == "pm_css":
            # same dimensions of the check matrices to emulate practically relevant instances and not make the problem too trivial
            rx = candidate_seed % (n - k + 1)
            if css_cnots:
                pair = NonPEqCodePairGenerator.css_codes_cnot_candidate(n, k, candidate_seed, rx=rx, gate_steps=GATE_STEPS)
            else:
                pair = NonPEqCodePairGenerator.css_codes_independent_candidate(n, k, candidate_seed, rx=rx)
        else:
            # keeps the two stabilizer codes related somehow to emulate practically relevant instances
            pair = NonPEqCodePairGenerator.stabilizer_codes_clifford_candidate(n, k, candidate_seed, gate_steps=GATE_STEPS)
        if certified_inequivalent(problem, pair, n, k):
            return pair
    raise RuntimeError(f"could not generate a certified {problem} negative for [[{n},{k}]], seed {seed}")


def invariant_matrices(problem: str, left: StabilizerCode, right: StabilizerCode) -> tuple:
    row_basis = p_stab._row_basis
    if problem == "pm_css":
        return row_basis(left.Hx), row_basis(left.Hz), row_basis(right.Hx), row_basis(right.Hz)
    return row_basis(left.symplectic), row_basis(right.symplectic)


def evaluate_invariant(name: str, problem: str, *matrices: Any) -> bool:
    if name == "linear_dependency" and problem == "pm_stb":
        return bool(p_stab.preserved_linear_dependencies(*matrices))
    if name == "linear_dependency" and problem == "pm_css":
        return bool(p_css.preserved_linear_dependencies(*matrices))
    if name == "signatures" and problem == "pm_stb":
        return bool(p_stab.preserved_punctured_hull_weight_enumerator(*matrices)[0])
    if name == "signatures" and problem == "pm_css":
        return bool(p_css.preserved_punctured_hull_weight_enumerator(*matrices)[0])
    if name == "local_invariant" and problem == "lc_stb":
        return bool(lc_stb.preserved_low_degree_local_invariant(*matrices))
    raise ValueError(f"unknown invariant {name!r} for {problem!r}")
