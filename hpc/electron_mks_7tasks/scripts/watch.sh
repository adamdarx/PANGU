#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
select_case "${1:?Missing case}"
load_environment
exec 10>"$RUN_DIR/monitor.lock"
flock -n 10 || exit 0
printf '%s\n' "$$" > "$RUN_DIR/monitor.pid"
echo "$(date -Is) monitoring $CASE"
while [[ ! -e "$RUN_DIR/STOP" && ! -e "$RUN_DIR/DONE" ]]; do
  job=$(<"$RUN_DIR/job.id")
  if ! active=$(squeue -h -j "$job" -o '%T'); then sleep "$POLL_SECONDS"; continue; fi
  if [[ -n "$active" ]]; then sleep "$POLL_SECONDS"; continue; fi
  if ! state=$(sacct -X -n -P -j "$job" --format=State | head -n 1); then
    sleep "$POLL_SECONDS"; continue
  fi
  state=${state%% *}; state=${state%%+*}
  case "$state" in
    COMPLETED|TIMEOUT|NODE_FAIL|PREEMPTED) ;;
    FAILED|OUT_OF_MEMORY|CANCELLED|BOOT_FAIL|DEADLINE)
      echo "$(date -Is) $job $state; stopped for inspection."; exit 1 ;;
    *) sleep "$POLL_SECONDS"; continue ;;
  esac
  exec 9>"$RUN_DIR/run.lock"
  flock 9
  [[ ! -e "$RUN_DIR/STOP" && ! -e "$RUN_DIR/DONE" ]] || exit 0
  if [[ "$(<"$RUN_DIR/job.id")" != "$job" ]]; then
    flock -u 9; sleep "$POLL_SECONDS"; continue
  fi
  "$PYTHON" "$BUNDLE/scripts/checkpoint.py" "$RUN_DIR" > "$RUN_DIR/checkpoint.monitor"
  [[ -s "$RUN_DIR/checkpoint.monitor" ]] || { echo 'No restart candidates; stopped.'; exit 1; }
  # Preflight and completion are checked on the GPU node, never from log maxima.
  enqueue "$job"
  flock -u 9
  sleep "$POLL_SECONDS"
done
