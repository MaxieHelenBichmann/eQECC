#!/usr/bin/env bash
# Run the fixed thesis invariant suite.
# Run from the repository root inside the uv environment.

set -euo pipefail

mkdir -p results

echo "Starting invariant benchmarks"
exec python3 -u -m benchmarks.thesis.thesis_invariants \
    --timeout 60 \
    --memory-limit "13GiB" \
    --verbose \
    --output "results/invariants.csv" \
    >"results/invariants.log" \
    2>"results/invariants.err"
