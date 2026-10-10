#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
select_case "${1:?Usage: submit.sh CASE}"
load_environment
[[ -x "$EXE" ]] || { echo "Build $EXE first" >&2; exit 1; }
[[ ! -e "$RUN_DIR/STOP" ]] || { echo 'STOP exists; remove it to resume.' >&2; exit 1; }
[[ ! -e "$RUN_DIR/DONE" ]] || { echo "$CASE already complete"; exit 0; }
exec 9>"$RUN_DIR/run.lock"
if ! flock -n 9; then start_monitor; exit 0; fi
if [[ -s "$RUN_DIR/job.id" ]]; then
  active=$(squeue -h -j "$(<"$RUN_DIR/job.id")" -o '%T')
  if [[ -n "$active" ]]; then
    echo "Retaining existing job: $active"
    start_monitor
    exit 0
  fi
fi
enqueue
start_monitor
