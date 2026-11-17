#!/usr/bin/env bash
set -uo pipefail
rank=${OMPI_COMM_WORLD_RANK:?}
interval=1
[[ ${AA_MODE:-admission} == formal ]] && interval=10
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits --loop="$interval" > "gpu-memory-$rank.csv" &
mem_pid=$!
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader --loop="$interval" > "gpu-apps-$rank.csv" &
app_pid=$!
"$@" &
exe_pid=$!
trap 'kill -TERM "$exe_pid" "$mem_pid" "$app_pid" 2>/dev/null || true' TERM INT
wait "$exe_pid"
rc=$?
kill "$mem_pid" "$app_pid" 2>/dev/null || true
wait "$mem_pid" "$app_pid" 2>/dev/null || true
exit "$rc"
