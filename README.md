# ***E******Q***uivalence Checking for ***Q***uantum ***E***rror ***C***orrection **C**odes

This repository benchmarks different approaches to equivalence checking for quantum error-correcting codes (QECCs) under different equivalence notions. It is part of the implementation for my Bachelor's thesis, "Automated Equivalence Checking of Stabilizer Codes", which contributes to [MQT QECC](https://github.com/munich-quantum-toolkit/qecc). The required infrastructure and code representation classes are taken from that project.

Additionally, it functions as a replication package for data and methodology in the paper "Methods for Automated Equivalence Checking of Stabilizer Codes", which is written on the basis of my thesis.

This README does not discuss the examined equivalence notions or their theoretical foundations in depth; those are covered in the thesis or paper. References for the implemented algorithms are cited in the corresponding source files.

This repository is currently not intended to be installed as a package. Its final hybrid algorithms are contributed to [MQT QECC](https://github.com/munich-quantum-toolkit/qecc).

## Problems

The following problems are benchmarked. They are expressed using the repository's input representation for a QECC $C$: a binary symplectic stabilizer matrix $\text{S}(C) \in \mathbb{F}_2^{r \times 2n}$, or the parity-check matrices $\text{H}_x(C) \in \mathbb{F}_2^{r_x \times n}$ and $\text{H}_z(C) \in \mathbb{F}_2^{r_z \times n}$ when the code is CSS.

### Permutation Equivalence

- **PM-STB**: Are two given stabilizer codes $C$ and $C'$ equivalent up to a permutation of the physical qubits?

$$
\exists P \in \mathfrak{S}_n,\ \exists R \in \mathrm{GL}(r, \mathbb{F}_2):
\qquad
\text{S}(C') =
R \cdot \text{S}(C) \cdot
\begin{bmatrix}
P & 0 \\
0 & P
\end{bmatrix}
$$

- **PM-CSS**: Are two given CSS codes $C$ and $C'$ equivalent up to a permutation of the physical qubits?

$$
\begin{aligned}
&\exists P \in \mathfrak{S}_n,\quad
\exists R_x \in \mathrm{GL}(r_x, \mathbb{F}_2),\quad
\exists R_z \in \mathrm{GL}(r_z, \mathbb{F}_2): \\
&\text{H}_x(C') = R_x \text{H}_x(C) P \enspace \text{ and } \enspace\text{H}_z(C') = R_z \cdot \text{H}_z(C) \cdot P
\end{aligned}
$$

### Local-Clifford Equivalence

- **LC-STB**: Are two given stabilizer codes $C$ and $C'$ local-Clifford equivalent, meaning that they define the same codespace up to local Clifford gates on the output qubits?

$$
\begin{aligned}
&\exists Q =
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
\in \text{Sp}(2n, \mathbb{F}_2), \quad
Q_i =
\begin{bmatrix}
a_{ii} & b_{ii} \\
c_{ii} & d_{ii}
\end{bmatrix}
\in \text{Sp}(2, \mathbb{F}_2), \quad
\exists R \in \mathrm{GL}(r, \mathbb{F}_2):\\
&\text{S}(C') = R \cdot \text{S}(C) \cdot Q
\end{aligned}
$$

- **LC-CSS**: Is a given stabilizer code $C$ local-Clifford equivalent to a CSS code?

$$
\begin{aligned}
&\exists C' \text{ with }
\enspace
\text{S}(C') =
\begin{bmatrix}
\text{H}_x(C') & 0 \\
0 & \text{H}_z(C')
\end{bmatrix}:\\
&\exists Q =
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
\in \text{Sp}(2n, \mathbb{F}_2), \quad
Q_i =
\begin{bmatrix}
a_{ii} & b_{ii} \\
c_{ii} & d_{ii}
\end{bmatrix}
\in \text{Sp}(2, \mathbb{F}_2), \quad
\exists R \in \mathrm{GL}(r, \mathbb{F}_2):\\
&\text{S}(C') = R \cdot \text{S}(C) \cdot Q
\end{aligned}
$$

## Approaches

### PM-STB

- [x] Brute-force Search
- [x] Classical Approaches
- [x] Automorphism Groups
- [x] Graph Isomorphism
- [x] SAT

### PM-CSS

- [x] Brute-force Search
- [x] Classical Approaches
- [x] Graph Isomorphism
- [x] Matroid Isomorphism
- [x] SAT

### LC-STB

- [x] Brute-force Search
- [x] Graph Isomorphism
- [x] Graph-state LSE **(only valid for $k < 2$ or with restrictions on logical operators)**
- [x] KLS normal form and LC orbit
- [x] SAT

### LC-CSS

- [x] Brute-force Search
- [x] KLS normal form and LC orbit
- [x] Clifford orbit
- [x] LC orbit **(only valid for $k < 2$ or with restrictions on logical operators)**
- [x] SAT

## Scope

Here, we measure the runtime of the Python algorithms on the expected workload: input codes $[[n,k,d]]$ with $n$ ranging from 2 to approximately 50, plus some larger structured cases.

This repository is not currently intended for detailed benchmarking or profiling analyses.[^1] The goal is to understand the algorithms' different complexity classes and make a more informed decision about the hybrid implementations.

Inputs are guaranteed to be valid and of same dimensions, so the prototype and hybrid algorithms in do not need to verify basic validity conditions.

[^1]: A future C++ implementation could support more rigorous benchmarks, but that is currently outside the scope of the thesis.

## Repository Structure

The repository is structured as a minimal local benchmark application.

```text
src/
  core/              # base QECC classes from MQT-QECC
    pauli.py
    symplectic.py
    stabilizer_code.py
    css_code.py
  algorithms/        # equivalence-checking implementations
    lc_css/
    lc_stb/
    p_css/
    p_stb/
  invariants/        # invariants under the equivalence relations
  hybrids/           # hybrid solutions combining best aspects of the algorithms
    lc_css.py
    lc_stb.py
    p_css.py
    p_stab.py

tests/               # partially randomized tests and edge-case tests
  hybrids/
  inv/
  lc_css/
  lc_stb/
  p_css/
  p_stab/
  bm/

benchmarks/
  thesis/                     # data collection scripts used for thesis
    thesis_prototypes.py
    thesis_hybrids.py
    thesis_invariants.py
  experiments/
    utils.py                  # randomization utilities
    run.py                    # one supervised function call
    statistics.py             # seeded repetitions, aggregation, and CSV append
    generators_random.py
    generators_structured.py
  run_hybrids.sh
  run_invariants.sh
  run_multiple.sh

data/                # structured case inputs

results/             # plotting tools; generated result artifacts are ignored by git

paper/               # replication package for the subsequent paper 
```

## Running the Thesis Benchmarks

The project is managed with [uv](https://docs.astral.sh/uv/). Create the environment with the exact locked dependencies first:

```bash
uv sync
```

This creates virtual environment, which every command below runs inside, either with `uv run <command>` or after `source .venv/bin/activate`.

The benchmark infrastructure offers resource-restriction and measurement infrastructure (`benchmarks.experiments.run.run(...)` `benchmarks.experiments.statistics.run_statistics(...)`). 

The thesis benchmarks (`benchmarks.thesis.thesis_prototypes`, `benchmarks.thesis.thesis_hybrids`, `benchmarks.thesis.thesis_invariants` ) are rather extensive and customizable, meant to provide a whole picture of the practical performance of prototype implementations, hybrids and invariants.

For example, running a random prototype benchmark from the repository root:

```bash
python3 -m benchmarks.thesis.thesis_prototypes \
  --algorithm pm_css_sat --nmin 5 --nmax 8 \
  --timeout 200 --memory-limit 4GiB \
  --nr-seeds 10 --output results/pm_css_sat_random.csv
```

### Tests, Linter, Formatter

[![Tests](https://github.com/MaxieHelenBichmann/bm_qecc/actions/workflows/tests.yml/badge.svg)](https://github.com/MaxieHelenBichmann/bm_qecc/actions/workflows/tests.yml)

The test suite is located in `tests/`. They include unit, regression and randomized tests.

```bash
uv run pytest
```

The linter, the formatter and the type checker for the replication package run the same way; `ruff format` rewrites files in place, `ruff format --check` only reports:

```bash
uv run ruff check
uv run ruff format
uv run mypy paper/
```

### Dependencies

Requires Linux or macOS with a C compiler and Python 3.13 or newer; Python 3.13 is the tested version in CI and for the paper replication package. The compiler is needed because `pynauty` is built from source on these versions; Windows is not supported by `pynauty` and by the benchmark supervision.

Apart from the Python dependencies declared in `pyproject.toml` and locked in `uv.lock`, the automorphism-group algorithm uses [GAP and Guava](https://docs.gap-system.org/pkg/guava/doc/manual.pdf), so a GAP executable and Guava's dependencies are required. Place the Guava dependencies in `bm_qecc/.gap` and set the path to the GAP executable before running this algorithm:

```bash
export GAP_EXECUTABLE=/path/to/gap
```

This approach is expected to be less efficient than the alternatives and is included primarily for comparative measurements.

### AI Transparency

While I used coding agents in my workflow, I kept that usage and especially AI-generated code to a minimum. 
It is limited to small/fine-grained library-specific functions, i.e. my helper functions calling certain library functions with some additional checks (see `_kernel_basis` or `_row_basis`). Broader-scope algorithms and algorithm components are written by me.  
Some basic tests are generated as well.
The only relevant files where I used AI more than that are the benchmark
infrastructure in `benchmarks/experiments/`, especially `run.py`,
`statistics.py`, and `utils.py`, plus `benchmarks/run_*.sh`, for infrastructure, resource
restrictions and instance generation (reviewed by me).
