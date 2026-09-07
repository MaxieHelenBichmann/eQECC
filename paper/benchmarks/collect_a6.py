"""Collect PM-STB SAT statistics on CSS inputs (A6).

Same positive and negative CSS pairs as the PM-CSS random suite, decided with
the tableau encoding. Appends one summary row per seeded batch to
pm_stb_sat_on_css.csv; re-running recomputes every batch and the extractor keeps
the latest row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.experiments.generators_random import PEqCodePairGenerator
from benchmarks.experiments.statistics import BenchmarkCase, run_statistics
from benchmarks.thesis.thesis_prototypes import DecisionAlgorithm, measurement_dimensions
from paper.benchmarks.common import (
    COLLECTED_DIR,
    MASTER_SEED,
    MEMORY_LIMIT_BYTES,
    TIMEOUT_SECONDS,
    certified_negative_pair,
)
from src.algorithms.p_stb.p_stab_sat import are_peq_stab_sat

NUM_SEEDS = 10
VERBOSE = True
N_RANGE = (3, 47)
OUTPUT_FILE = COLLECTED_DIR / "pm_stb_sat_on_css.csv"


@dataclass(frozen=True)
class CSSCaseGenerator:
    n: int
    k: int
    positive: bool

    seed_upper_bound = 1_000

    @property
    def __name__(self) -> str:
        label = "positive" if self.positive else "negative"
        return f"random_pm_stb_sat_on_css_{self.n}_{self.k}_{label}"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": "pm_stb_sat_on_css",
            "generator": self.__name__,
            "n": self.n,
            "k": self.k,
            "positive": self.positive,
        }

    def __call__(self, seed: int) -> BenchmarkCase:
        if self.positive:
            pair = PEqCodePairGenerator.css_codes_basis_changed(self.n, self.k, seed)
        else:
            pair = certified_negative_pair("pm_css", self.n, self.k, seed)
        return BenchmarkCase(tuple(pair), self.positive, self.metadata)


def collect() -> None:
    algorithm = DecisionAlgorithm("pm_stb_sat_on_css", are_peq_stab_sat)
    for n, k in measurement_dimensions(*N_RANGE):
        for positive in (True, False):
            if VERBOSE:
                print(f"pm_stb_sat_on_css [[{n},{k}]] {'positive' if positive else 'negative'}", flush=True)
            run_statistics(
                algorithm,
                CSSCaseGenerator(n, k, positive),
                MASTER_SEED,
                NUM_SEEDS,
                OUTPUT_FILE,
                timeout=TIMEOUT_SECONDS,
                max_memory_bytes=MEMORY_LIMIT_BYTES,
                verbose=VERBOSE,
            )


if __name__ == "__main__":
    collect()
