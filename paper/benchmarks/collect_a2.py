"""Collect signature partition sizes of random codes (A2).

One random stabilizer or CSS code per (n, k, seed); the instance result stores the
sizes of its signature classes and q = sum(|J_i|^2) / n^2, the probability that two
qubits drawn with replacement share a signature. Rows are appended to
signature_space.csv and existing keys are skipped on restart.
"""

from __future__ import annotations

import numpy as np

from benchmarks.experiments.run import run
from benchmarks.experiments.statistics import deterministic_seeds
from benchmarks.experiments.utils import random_css_code, random_stabilizer_code
from benchmarks.thesis.thesis_prototypes import measurement_dimensions
from paper.benchmarks.common import (
    COLLECTED_DIR,
    MASTER_SEED,
    MEMORY_LIMIT_BYTES,
    TIMEOUT_SECONDS,
    append_row,
    completed_keys,
    execution_status,
)
from src.hybrids import p_css, p_stab

NUM_SEEDS = 10
SEEDS = deterministic_seeds(MASTER_SEED, NUM_SEEDS, upper_bound=1_000)
# ascending rank, so the cheap cells of every n come first
DIMENSIONS = tuple(sorted(measurement_dimensions(), key=lambda nk: (nk[0] - nk[1], nk[0], nk[1])))
OUTPUT_FILE = COLLECTED_DIR / "signature_space.csv"
PROBLEMS = ("pm_stb", "pm_css")
KEY_FIELDS = ("problem", "n", "k", "seed")
FIELDS = (
    "problem",
    "seed",
    "n",
    "k",
    "r",
    "x_rank",
    "class_sizes",
    "q_pairs",
    "status",
    "timeout",
    "memory_limited",
    "error",
)


def generate_random_code(problem: str, n: int, k: int, seed: int):
    if problem == "pm_css":
        rng = np.random.default_rng(seed)
        x_rank = int(rng.integers(0, n - k + 1))
        code_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        return random_css_code(n, k, rx=x_rank, seed=code_seed), x_rank
    return random_stabilizer_code(n, k, seed=seed), None


def evaluate_signature_partition(problem: str, code) -> list[int]:
    row_basis = p_stab._row_basis
    if problem == "pm_css":
        hx, hz = row_basis(code.Hx), row_basis(code.Hz)
        compatible, partition, _ = p_css.preserved_punctured_hull_weight_enumerator(hx, hz, hx, hz)
    else:
        matrix = row_basis(code.symplectic)
        compatible, partition, _ = p_stab.preserved_punctured_hull_weight_enumerator(matrix, matrix)
    if not compatible or partition is None:
        raise RuntimeError("a code was unexpectedly incompatible with itself")
    return sorted((len(group) for group in partition.values()), reverse=True)


def signature_metric(class_sizes, n: int) -> float:
    return sum(size * size for size in class_sizes) / (n * n)


def collect(dimensions=DIMENSIONS, seeds=SEEDS, output_file=OUTPUT_FILE) -> list[dict]:
    completed = completed_keys(output_file, KEY_FIELDS)
    rows = []
    for problem in PROBLEMS:
        print(f"signature partitions: {problem}", flush=True)
        for n, k in dimensions:
            for seed in seeds:
                key = (problem, str(n), str(k), str(seed))
                if key in completed:
                    continue
                row = {"problem": problem, "seed": seed, "n": n, "k": k, "r": n - k, "x_rank": ""}
                try:
                    code, x_rank = generate_random_code(problem, n, k, seed)
                    row["x_rank"] = "" if x_rank is None else x_rank
                except Exception as exc:
                    row.update(
                        class_sizes="",
                        q_pairs=None,
                        status="generation_error",
                        timeout=False,
                        memory_limited=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    result = run(
                        evaluate_signature_partition,
                        (problem, code),
                        None,
                        timeout=TIMEOUT_SECONDS,
                        max_memory_bytes=MEMORY_LIMIT_BYTES,
                    )
                    status = execution_status(result)
                    sizes = list(result.result) if status == "success" else []
                    row.update(
                        class_sizes=" ".join(map(str, sizes)),
                        q_pairs=signature_metric(sizes, n) if status == "success" else None,
                        status=status,
                        timeout=result.timed_out,
                        memory_limited=result.memory_exceeded,
                        error=result.error or "",
                    )
                append_row(output_file, row, FIELDS)
                rows.append(row)
                completed.add(key)
    print(f"appended {len(rows)} rows to {output_file}", flush=True)
    return rows


if __name__ == "__main__":
    collect()
