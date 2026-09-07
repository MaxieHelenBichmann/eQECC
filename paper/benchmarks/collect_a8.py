"""Collect runtimes and deciding stages of the paper hybrids on named codes (A8).

    python3 -m paper.benchmarks.collect_a8 [--problem pm_stb|pm_css|lc_stb ...]

Per problem, code, label, and seed one instance is generated and cached in
hybrids/<problem>_instances.csv, then the hybrid runs on it under supervision
and hybrids/<problem>_raw.csv receives its status, runtime, the stage that
decided, and for a killed call the stage it was stuck in. Positives pair the
code with an equivalent presentation (random permutation or local Clifford plus
a generator basis change); negatives follow A1 with two random Clifford gates or
CNOTs, certified by SAT, matroid isomorphism, or a permutation-invariant
mismatch for the largest CSS codes. Restarting skips keys present in the raw
file and reuses cached instances.
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.experiments.generators_random import _certificate as css_certificate
from benchmarks.experiments.generators_structured import (
    LCEqCodePairGenerator,
    NonPEqCodePairGenerator,
    PEqCodePairGenerator,
)
from benchmarks.experiments.run import run
from benchmarks.experiments.statistics import deterministic_seeds
from paper.hybrids.lc_stb import are_lceq
from paper.hybrids.pm_css import are_peq_css
from paper.hybrids.pm_stb import are_peq_stab
from paper.benchmarks.common import (
    CERTIFIERS,
    COLLECTED_DIR,
    GATE_STEPS,
    MASTER_SEED,
    MEMORY_LIMIT_BYTES,
    TIMEOUT_SECONDS,
    append_row,
    css_certifier,
    execution_status,
    read_rows,
)
from src.core.css_code import CSSCode
from src.core.pauli import StabilizerTableau
from src.core.stabilizer_code import StabilizerCode

OUTPUT_DIRECTORY = COLLECTED_DIR / "hybrids"

CODES = (
    "bell",
    "3q_rep",
    "5q_prf",
    "steane",
    "shor",
    "carbon",
    "hamming_15",
    "15q_optimal",
    "tetrahedral",
    "golay",
    "rot_surf_d5",
    "hamming_31",
    "coco_488",
    "coco_666",
    "bb_72",
    "bb_90",
    "bb_108",
    "bb_144",
)
NON_CSS_CODES = frozenset({"5q_prf", "15q_optimal"})
HYBRIDS = {
    "pm_stb": are_peq_stab,
    "pm_css": are_peq_css,
    "lc_stb": are_lceq,
}

NUM_SEEDS = 10
SEEDS = deterministic_seeds(MASTER_SEED, NUM_SEEDS, upper_bound=1_000)
NEGATIVE_MAX_ATTEMPTS = 1_000
GENERATION_TIMEOUT_SECONDS = 900.0
VERBOSE = True

# stage tags the hybrids print on entry, in pipeline order
STAGES = ("CI", "EI", "S", "BF", "MI", "GI", "SAT", "LSE")
TRIVIAL = "trivial"
UNREACHED = "start"
DECIDED_MARKER = "#decided_by "

KEY_FIELDS = ("code", "positive", "seed")
INSTANCE_FIELDS = (*KEY_FIELDS, "n", "k", "status", "left", "right", "error")
RAW_FIELDS = (
    "problem",
    *KEY_FIELDS,
    "n",
    "k",
    "status",
    "runtime_seconds",
    "decided_by",
    "stuck_at",
    "timeout_seconds",
    "error",
)


# CSV persistence -------------------------------------------------------------------------------


def row_key(row) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def encode_code(code: StabilizerCode) -> str:
    # css:<Hx rows>|<Hz rows> or stb:<symplectic rows>, one bit string per row
    def rows(matrix: np.ndarray) -> str:
        return "/".join("".join(str(int(bit)) for bit in row) for row in np.asarray(matrix) % 2)

    if isinstance(code, CSSCode):
        return f"css:{rows(code.Hx)}|{rows(code.Hz)}"
    return f"stb:{rows(code.symplectic)}"


def decode_code(text: str, n: int) -> StabilizerCode:
    def matrix(part: str, columns: int) -> np.ndarray:
        rows = [[int(bit) for bit in row] for row in part.split("/") if row]
        return np.array(rows, dtype=np.int8).reshape(-1, columns)

    kind, body = text.split(":", 1)
    if kind == "css":
        hx, hz = body.split("|")
        return CSSCode(matrix(hx, n), matrix(hz, n), n=n)
    return StabilizerCode(StabilizerTableau(matrix(body, 2 * n)), n=n)


# Instance generation ---------------------------------------------------------------------------


def certified_inequivalent(problem: str, left: StabilizerCode, right: StabilizerCode) -> bool:
    if problem == "pm_css":
        certifier = css_certifier(left.n, left.k)
        if certifier is None:
            return css_certificate(left) != css_certificate(right)
    else:
        certifier = CERTIFIERS[problem]
    return not certifier(left, right)


def generate_pair(problem: str, code_name: str, positive: bool, seed: int) -> tuple[StabilizerCode, StabilizerCode]:
    if positive:
        if problem == "lc_stb":
            return LCEqCodePairGenerator.stabilizer_codes_local_clifford(code_name, seed)
        if code_name in NON_CSS_CODES:
            return PEqCodePairGenerator.stabilizer_codes_basis_changed(code_name, seed)
        # PM-STB and PM-CSS receive the identical pair for a CSS code.
        return PEqCodePairGenerator.css_codes_basis_changed(code_name, seed)

    for attempt in range(NEGATIVE_MAX_ATTEMPTS):
        attempt_seed = seed * NEGATIVE_MAX_ATTEMPTS + attempt
        if problem == "pm_css":
            code, candidate = NonPEqCodePairGenerator.css_codes_cnot_candidate(
                code_name, attempt_seed, gate_steps=GATE_STEPS
            )
        else:
            code, candidate = NonPEqCodePairGenerator.stabilizer_codes_clifford_candidate(
                code_name, attempt_seed, gate_steps=GATE_STEPS
            )
        if certified_inequivalent(problem, code, candidate):
            return code, candidate
    raise RuntimeError(f"no certified {problem} negative for {code_name}, seed {seed}")


def generate_instance(problem: str, code_name: str, positive: bool, seed: int) -> dict:
    left, right = generate_pair(problem, code_name, positive, seed)
    return {
        "code": code_name,
        "positive": positive,
        "seed": seed,
        "n": left.n,
        "k": left.k,
        "status": "success",
        "left": encode_code(left),
        "right": encode_code(right),
        "error": "",
    }


# Supervised hybrid execution -------------------------------------------------------------------


@dataclass(frozen=True)
class TracedHybrid:
    """Run a hybrid with its stage tags written to a line-buffered log, so they survive a kill."""

    function: Callable
    log_path: str

    def __call__(self, left: StabilizerCode, right: StabilizerCode) -> bool:
        with open(self.log_path, "w", buffering=1, encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                decision, stage = self.function(left, right)
                print(f"{DECIDED_MARKER}{stage or TRIVIAL}")
        return bool(decision)


def read_trace(log_path: Path) -> tuple[list[str], str]:
    trace, decided_by = [], ""
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line in STAGES:
            trace.append(line)
        elif line.startswith(DECIDED_MARKER):
            decided_by = line[len(DECIDED_MARKER) :]
    return trace, decided_by


def run_instance(problem: str, instance, log_path: Path, *, timeout: float, memory_limit_bytes: int) -> dict:
    n = int(instance["n"])
    left, right = decode_code(instance["left"], n), decode_code(instance["right"], n)
    positive = instance["positive"] == "True"
    log_path.write_text("", encoding="utf-8")
    result = run(
        TracedHybrid(HYBRIDS[problem], str(log_path)),
        (left, right),
        positive,
        timeout=timeout,
        max_memory_bytes=memory_limit_bytes,
    )
    status = execution_status(result)
    if status == "success" and not result.result_is_expected:
        status = "unexpected"
    trace, decided_by = read_trace(log_path)
    killed = status in {"timeout", "memory_limited"}
    return {
        "problem": problem,
        "code": instance["code"],
        "positive": positive,
        "seed": instance["seed"],
        "n": n,
        "k": instance["k"],
        "status": status,
        "runtime_seconds": f"{result.runtime:.9f}",
        "decided_by": decided_by,
        "stuck_at": (trace[-1] if trace else UNREACHED) if killed else "",
        "timeout_seconds": timeout,
        "error": result.error or "",
    }


# Collection ------------------------------------------------------------------------------------


def collect(
    problems=tuple(HYBRIDS),
    *,
    codes=CODES,
    seeds=SEEDS,
    output_directory=OUTPUT_DIRECTORY,
    generation_timeout=GENERATION_TIMEOUT_SECONDS,
    timeout=TIMEOUT_SECONDS,
    memory_limit_bytes=MEMORY_LIMIT_BYTES,
    verbose=VERBOSE,
) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        log_path = Path(scratch) / "trace.log"
        for problem in problems:
            instances_file = output_directory / f"{problem}_instances.csv"
            raw_file = output_directory / f"{problem}_raw.csv"
            instances = {row_key(row): row for row in read_rows(instances_file)}
            measured = {row_key(row) for row in read_rows(raw_file)}
            for code_name in codes:
                if problem == "pm_css" and code_name in NON_CSS_CODES:
                    continue
                for positive in (True, False):
                    for seed in seeds:
                        key = (code_name, str(positive), str(seed))
                        if key in measured:
                            continue
                        instance = instances.get(key)
                        if instance is None:
                            outcome = run(
                                generate_instance,
                                (problem, code_name, positive, seed),
                                None,
                                timeout=generation_timeout,
                                max_memory_bytes=memory_limit_bytes,
                            )
                            instance = outcome.result or {
                                "code": code_name,
                                "positive": positive,
                                "seed": seed,
                                "n": "",
                                "k": "",
                                "status": "generation_error",
                                "left": "",
                                "right": "",
                                "error": outcome.error or f"generation {execution_status(outcome)}",
                            }
                            instance = {field: str(instance[field]) for field in INSTANCE_FIELDS}
                            append_row(instances_file, instance, INSTANCE_FIELDS)
                            instances[key] = instance
                        if instance["status"] != "success":
                            row = {
                                "problem": problem,
                                "code": code_name,
                                "positive": positive,
                                "seed": seed,
                                "n": instance["n"],
                                "k": instance["k"],
                                "status": "generation_error",
                                "runtime_seconds": "",
                                "decided_by": "",
                                "stuck_at": "",
                                "timeout_seconds": timeout,
                                "error": instance["error"],
                            }
                        else:
                            row = run_instance(
                                problem, instance, log_path, timeout=timeout, memory_limit_bytes=memory_limit_bytes
                            )
                        append_row(raw_file, row, RAW_FIELDS)
                        measured.add(key)
                        if verbose:
                            detail = row["decided_by"] or row["stuck_at"] or row["error"]
                            runtime = f"{row['runtime_seconds']}s " if row["runtime_seconds"] else ""
                            print(
                                f"{problem} {code_name} positive={positive} seed={seed}: "
                                f"{row['status']} {runtime}{detail}",
                                flush=True,
                            )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        choices=tuple(HYBRIDS),
        action="append",
        help="collect only this problem (repeatable); default: all",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    collect(parse_args().problem or tuple(HYBRIDS))
