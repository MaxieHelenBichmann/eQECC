#!/usr/bin/env bash
# Run the four thesis hybrids on the named structured codes, two at a time.
# Run from the repository root inside the uv environment.

set -u

# inclusive bounds
nmin=2
nmax=144

max_avail_mem_gib=26
max_jobs=2
memory_limit_gib=$((max_avail_mem_gib / max_jobs))
timeout_seconds=5400

algorithms=(
    pm_css_hybrid
    pm_stb_hybrid
    lc_stb_hybrid
    lc_css_hybrid
)

output_dir="results/hybrids"
status=0
pids=()

mkdir -p "$output_dir"

run_algo() {
    local algorithm_name="$1"
    local output_base="${output_dir}/${algorithm_name}_structured"

    echo "Starting ${algorithm_name}"
    exec python3 -u -m benchmarks.thesis.thesis_hybrids \
        --algorithm "$algorithm_name" \
        --nmin "$nmin" \
        --nmax "$nmax" \
        --timeout "$timeout_seconds" \
        --memory-limit "${memory_limit_gib}GiB" \
        --verbose \
        --output "${output_base}.csv" \
        >"${output_base}.log" \
        2>"${output_base}.err"
}

wait_for_one() {
    local finished_pid
    local wait_status=0
    local pid
    local remaining_pids=()

    wait -n -p finished_pid "${pids[@]}" || wait_status=$?
    if ((wait_status != 0 && status == 0)); then
        status=$wait_status
    fi

    for pid in "${pids[@]}"; do
        if [[ "$pid" != "$finished_pid" ]]; then
            remaining_pids+=("$pid")
        fi
    done
    pids=("${remaining_pids[@]}")
}

for algorithm_name in "${algorithms[@]}"; do
    while ((${#pids[@]} >= max_jobs)); do
        wait_for_one
    done

    run_algo "$algorithm_name" &
    pids+=("$!")
done

while ((${#pids[@]} > 0)); do
    wait_for_one
done

exit "$status"
