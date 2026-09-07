"""Collect invariant decisions on certified inequivalent code pairs (A1).

Negative instances are a random source code and a perturbed copy: two random Clifford
gates for PM-STB and LC-STB, two random CNOTs preserving the CSS form and both
check ranks for PM-CSS. A candidate is kept once SAT, or matroid isomorphism
for CSS codes with r > 9, proves inequivalence; CSS codes beyond both certifiers
use the cascaded generator. Rows are appended to invariant_rejections.csv and
existing keys are skipped on restart.
"""

from __future__ import annotations

from benchmarks.experiments.run import run
from benchmarks.experiments.statistics import deterministic_seeds
from benchmarks.thesis.thesis_prototypes import measurement_dimensions
from paper.benchmarks.common import (
    COLLECTED_DIR,
    INVARIANTS,
    MASTER_SEED,
    MEMORY_LIMIT_BYTES,
    TIMEOUT_SECONDS,
    append_row,
    certified_negative_pair,
    completed_keys,
    evaluate_invariant,
    execution_status,
    invariant_matrices,
)

NUM_SEEDS = 10
SEEDS = deterministic_seeds(MASTER_SEED, NUM_SEEDS, upper_bound=1_000)
DIMENSIONS = tuple(measurement_dimensions())
OUTPUT_FILE = COLLECTED_DIR / "invariant_rejections.csv"
PROBLEMS = ("pm_stb", "pm_css", "lc_stb")
KEY_FIELDS = ("problem", "n", "k", "seed", "invariant")
FIELDS = (
    "problem",
    "instance_id",
    "seed",
    "n",
    "k",
    "r",
    "invariant",
    "rejected",
    "status",
    "timeout",
    "memory_limited",
    "error",
)


def collect(dimensions=DIMENSIONS, seeds=SEEDS, output_file=OUTPUT_FILE) -> list[dict]:
    completed = completed_keys(output_file, KEY_FIELDS)
    rows = []
    for problem in PROBLEMS:
        print(f"invariant rejections: {problem}", flush=True)
        for n, k in dimensions:
            for seed in seeds:
                missing = [
                    invariant
                    for invariant in INVARIANTS[problem]
                    if (problem, str(n), str(k), str(seed), invariant) not in completed
                ]
                if not missing:
                    continue
                base = {
                    "problem": problem,
                    "instance_id": f"{problem}-n{n}k{k}-s{seed}",
                    "seed": seed,
                    "n": n,
                    "k": k,
                    "r": n - k,
                }
                try:
                    pair = certified_negative_pair(problem, n, k, seed, css_cnots=True)
                    matrices = invariant_matrices(problem, *pair)
                except Exception as exc:
                    matrices = None
                    error = f"{type(exc).__name__}: {exc}"
                for invariant in missing:
                    row = {**base, "invariant": invariant}
                    if matrices is None:
                        row.update(
                            rejected=None, status="generation_error", timeout=False, memory_limited=False, error=error
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
                            rejected=not result.result if status == "success" else None,
                            status=status,
                            timeout=result.timed_out,
                            memory_limited=result.memory_exceeded,
                            error=result.error or "",
                        )
                    append_row(output_file, row, FIELDS)
                    rows.append(row)
                    completed.add((problem, str(n), str(k), str(seed), invariant))
    print(f"appended {len(rows)} rows to {output_file}", flush=True)
    return rows


if __name__ == "__main__":
    collect()
