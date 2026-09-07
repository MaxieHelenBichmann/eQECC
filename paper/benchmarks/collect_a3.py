"""Collect invariant runtimes (A3).

5 positive and 5 negative pairs per (n, k). Negative instances are certified as
in A1, except that PM-CSS negatives are two independent codes with matching
check ranks. Only the invariant call is timed; the row-basis reduction of the
inputs happens before. Rows are appended to invariant_timings.csv and existing
keys are skipped on restart.
"""

from __future__ import annotations

from benchmarks.experiments.run import run
from benchmarks.experiments.statistics import deterministic_seeds
from benchmarks.thesis.thesis_prototypes import RandomCaseGenerator, measurement_dimensions
from paper.benchmarks.common import (
    COLLECTED_DIR,
    INVARIANTS,
    MASTER_SEED,
    MEMORY_LIMIT_BYTES,
    TIMEOUT_SECONDS,
    CodePair,
    append_row,
    certified_negative_pair,
    completed_keys,
    evaluate_invariant,
    execution_status,
    invariant_matrices,
)

NUM_SEEDS = 5
SEEDS = deterministic_seeds(MASTER_SEED, NUM_SEEDS, upper_bound=1_000)
DIMENSIONS = tuple(measurement_dimensions())
VERBOSE = True
OUTPUT_FILE = COLLECTED_DIR / "invariant_timings.csv"
KEY_FIELDS = ("problem", "invariant", "n", "k", "positive", "seed")
FIELDS = (
    "problem",
    "invariant",
    "instance_id",
    "seed",
    "n",
    "k",
    "r",
    "positive",
    "accepted",
    "runtime_seconds",
    "status",
    "timeout",
    "memory_limited",
    "error",
)


def generate_pair(problem: str, n: int, k: int, positive: bool, seed: int) -> CodePair:
    if positive:
        return RandomCaseGenerator(f"{problem}_sat", n, k, True)(seed).inputs
    return certified_negative_pair(problem, n, k, seed)


def collect(dimensions=DIMENSIONS, seeds=SEEDS, output_file=OUTPUT_FILE) -> list[dict]:
    completed = completed_keys(output_file, KEY_FIELDS)
    rows = []
    for problem, invariants in INVARIANTS.items():
        print(f"invariant timings: {problem}", flush=True)
        for n, k in dimensions:
            for positive in (True, False):
                label = "positive" if positive else "negative"
                for seed in seeds:
                    missing = [
                        invariant
                        for invariant in invariants
                        if (problem, invariant, str(n), str(k), str(positive), str(seed)) not in completed
                    ]
                    if not missing:
                        continue
                    if VERBOSE:
                        print(f"    [[{n},{k}]] {label} seed={seed}", flush=True)
                    base = {
                        "problem": problem,
                        "instance_id": f"{problem}-n{n}k{k}-s{seed}-{label}",
                        "seed": seed,
                        "n": n,
                        "k": k,
                        "r": n - k,
                        "positive": positive,
                    }
                    try:
                        matrices = invariant_matrices(problem, *generate_pair(problem, n, k, positive, seed))
                    except Exception as exc:
                        matrices = None
                        error = f"{type(exc).__name__}: {exc}"
                    for invariant in missing:
                        row = {**base, "invariant": invariant}
                        if matrices is None:
                            row.update(
                                accepted=None,
                                runtime_seconds=None,
                                status="generation_error",
                                timeout=False,
                                memory_limited=False,
                                error=error,
                            )
                        else:
                            result = run(
                                evaluate_invariant,
                                (invariant, problem, *matrices),
                                None,
                                timeout=TIMEOUT_SECONDS,
                                max_memory_bytes=MEMORY_LIMIT_BYTES,
                            )
                            status = execution_status(result)
                            row.update(
                                accepted=result.result if status == "success" else None,
                                runtime_seconds=result.runtime,
                                status=status,
                                timeout=result.timed_out,
                                memory_limited=result.memory_exceeded,
                                error=result.error or "",
                            )
                        append_row(output_file, row, FIELDS)
                        rows.append(row)
                        completed.add((problem, invariant, str(n), str(k), str(positive), str(seed)))
    print(f"appended {len(rows)} rows to {output_file}", flush=True)
    return rows


if __name__ == "__main__":
    collect()
