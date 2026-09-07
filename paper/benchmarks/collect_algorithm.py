"""Collect the random-suite statistics of the exact algorithms (A3 to A6).

    python3 -m paper.benchmarks.collect_algorithm [--algorithm SELECTOR ...]

Each selected algorithm appends to algorithms/<algorithm>.csv, one summary row
per seeded batch of positive or negative pairs; the extractors keep the latest
row per batch. Positives come from the thesis random suite, negatives are
certified as in A1, except that PM-CSS negatives are two independent codes with
matching check ranks. Generation and certification happen before the timed call.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from typing import Any

from benchmarks.experiments.statistics import BenchmarkCase, run_statistics
from benchmarks.thesis import resolve_names
from benchmarks.thesis.thesis_prototypes import ALGORITHMS, RandomCaseGenerator, measurement_dimensions
from paper.benchmarks.common import (
    COLLECTED_DIR,
    MASTER_SEED,
    MEMORY_LIMIT_BYTES,
    TIMEOUT_SECONDS,
    certified_negative_pair,
)
from paper.experiments.common import problem_for_algorithm

NUM_SEEDS = 10
VERBOSE = True
OUTPUT_DIRECTORY = COLLECTED_DIR / "algorithms"

# inclusive n ranges
ALGORITHM_N_RANGES = {
    "pm_stb_aut": (3, 13),
    "pm_stb_bruteforce": (3, 47),
    "pm_stb_classical": (3, 47),
    "pm_stb_graph_iso": (3, 47),
    "pm_stb_sat": (3, 47),
    "pm_css_bruteforce": (3, 47),
    "pm_css_classical": (3, 47),
    "pm_css_matroid": (3, 47),
    "pm_css_sat": (3, 47),
    "lc_stb_lse": (3, 47),
    "lc_stb_bruteforce": (3, 47),
    "lc_stb_graph_iso": (3, 47),
    "lc_stb_kls": (3, 47),
    "lc_stb_sat": (3, 47),
}
PAPER_ALGORITHMS = {name: ALGORITHMS[name] for name in ALGORITHM_N_RANGES}


@dataclass(frozen=True)
class CertifiedRandomCaseGenerator:
    """Thesis random cases, with the negatives replaced by certified pairs."""

    algorithm_name: str
    n: int
    k: int
    positive: bool

    seed_upper_bound = 1_000

    @property
    def __name__(self) -> str:
        label = "positive" if self.positive else "certified_negative"
        return f"random_{self.algorithm_name}_{self.n}_{self.k}_{label}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "generator": self.__name__,
            "name": None,
            "n": self.n,
            "k": self.k,
            "positive": self.positive,
            "density": None,
            "symmetry": None,
        }

    def __call__(self, seed: int) -> BenchmarkCase:
        if self.positive:
            inputs = RandomCaseGenerator(self.algorithm_name, self.n, self.k, True)(seed).inputs
        else:
            inputs = certified_negative_pair(problem_for_algorithm(self.algorithm_name), self.n, self.k, seed)
        return BenchmarkCase(tuple(inputs), self.positive, self.metadata)


def collect(algorithm_names) -> None:
    for algorithm_name in algorithm_names:
        if algorithm_name == "pm_stb_aut":
            gap_executable = os.environ.get("GAP_EXECUTABLE", "gap")
            if shutil.which(gap_executable) is None:
                print(
                    f"warning: skipping pm_stb_aut, GAP executable {gap_executable!r} not found "
                    "(install GAP with Guava or set GAP_EXECUTABLE)",
                    flush=True,
                )
                continue
        nmin, nmax = ALGORITHM_N_RANGES[algorithm_name]
        output_file = OUTPUT_DIRECTORY / f"{algorithm_name}.csv"
        if VERBOSE:
            print(f"{algorithm_name}: n={nmin}..{nmax}, {NUM_SEEDS} seeds/cell -> {output_file}", flush=True)
        for n, k in measurement_dimensions(nmin, nmax):
            if algorithm_name == "lc_stb_lse" and k >= 2:
                continue
            for positive in (True, False):
                if VERBOSE:
                    print(f"    [[{n},{k}]] {'positive' if positive else 'negative'}", flush=True)
                run_statistics(
                    ALGORITHMS[algorithm_name],
                    CertifiedRandomCaseGenerator(algorithm_name, n, k, positive),
                    MASTER_SEED,
                    NUM_SEEDS,
                    output_file,
                    timeout=TIMEOUT_SECONDS,
                    max_memory_bytes=MEMORY_LIMIT_BYTES,
                    verbose=VERBOSE,
                )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        action="append",
        metavar="SELECTOR",
        help="exact name, shell wildcard, or regex; repeatable; default: all",
    )
    args = parser.parse_args(argv)
    if args.algorithm is None:
        args.algorithm = list(ALGORITHM_N_RANGES)
    else:
        try:
            args.algorithm = resolve_names(args.algorithm, PAPER_ALGORITHMS)
        except ValueError as exc:
            parser.error(str(exc))
    return args


if __name__ == "__main__":
    collect(parse_args().algorithm)
