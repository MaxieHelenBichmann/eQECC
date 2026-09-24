#!/usr/bin/env bash
# Run every prototype algorithm of one problem family on the random thesis
# suite, two at a time. Run from the repository root inside the uv environment.

set -u

problem_type="pm_stb"

max_avail_mem_gib=26
max_jobs=2
memory_limit_gib=$((max_avail_mem_gib / max_jobs))
timeout_seconds=5400
status=0
pids=()

# inclusive bounds
n_range_for() {
    case "$1" in
        *) echo "" ;;
    esac
}

mkdir -p results

case "$problem_type" in
    pm_css)
        algorithms=(sat matroid graph_iso classical bruteforce)
        ;;
    pm_stb)
        algorithms=(sat graph_iso classical bruteforce aut)
        ;;
    lc_stb)
        algorithms=(sat kls lse graph_iso bruteforce)
        ;;
    lc_css)
        algorithms=(sat kls cliff_orbit lc_orbit bruteforce)
        ;;
    *)
        echo "unknown problem type: ${problem_type}" >&2
        exit 2
        ;;
esac

run_algo() {
    local algo="$1"
    local algorithm_name="${problem_type}_${algo}"
    local output_base="results/${problem_type}_${algo}_rdm"
    local n_args=()
    local n_range

    n_range="$(n_range_for "$algorithm_name")"
    if [[ -n "$n_range" ]]; then
        # shellcheck disable=SC2206
        n_args=(--nmin ${n_range% *} --nmax ${n_range#* })
    fi

    echo "Starting ${algorithm_name}"
    exec python3 -u -m benchmarks.thesis.thesis_prototypes \
        --algorithm "${algorithm_name}" \
        --timeout "$timeout_seconds" \
        --memory-limit "${memory_limit_gib}GiB" \
        --verbose \
        "${n_args[@]}" \
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
    if (( wait_status != 0 && status == 0 )); then
        status=$wait_status
    fi

    for pid in "${pids[@]}"; do
        if [[ "$pid" != "$finished_pid" ]]; then
            remaining_pids+=("$pid")
        fi
    done
    pids=("${remaining_pids[@]}")
}

for algo in "${algorithms[@]}"; do
    while (( ${#pids[@]} >= max_jobs )); do
        wait_for_one
    done

    run_algo "$algo" &
    pids+=("$!")
done

while (( ${#pids[@]} > 0 )); do
    wait_for_one
done

exit "$status"
