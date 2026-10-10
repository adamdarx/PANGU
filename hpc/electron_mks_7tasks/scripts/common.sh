#!/usr/bin/env bash
set -euo pipefail
BUNDLE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$BUNDLE/server.conf"
REPO=$(realpath -m -- "$REPO")
BUILD_DIR="$REPO/build-electron-mks"
EXE="$BUILD_DIR/src/pangu"
TARGET=30000
cases=(sane_a0 sane_a05 sane_a09 sane_a09375 mad_am09375 mad_a0 mad_ap09375)
select_case() {
  CASE=${1:?Specify a case name}
  local found=false item
  for item in "${cases[@]}"; do [[ "$CASE" != "$item" ]] || found=true; done
  "$found" || { echo "Unknown case: $CASE" >&2; return 2; }
  INPUT="$BUNDLE/cases/$CASE.in"
  RUN_DIR="$BUNDLE/run/$CASE"
  mkdir -p "$RUN_DIR/logs"
}
enqueue() {
  local predecessor=${1:-} job
  export REPO
  job=$(cd "$BUNDLE" && sbatch --parsable --job-name="E6_$CASE" \
    --partition="$PARTITION" --qos="$QOS" --nodes=1 --ntasks-per-node=2 \
    --ntasks=2 --cpus-per-task=1 --gres=gpu:2 --time="${WALL_HOURS}:00:00" \
    --chdir="$BUNDLE" --export=ALL --output="$RUN_DIR/logs/slurm-%j.out" \
    "$BUNDLE/scripts/job.sh" "$CASE" "$predecessor" 9>&- 10>&-)
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: $job" >&2; return 1; }
  printf '%s\n' "$job" > "$RUN_DIR/job.id.tmp"
  mv "$RUN_DIR/job.id.tmp" "$RUN_DIR/job.id"
  printf '%s %s predecessor=%s\n' "$(date -Is)" "$job" "$predecessor" >> "$RUN_DIR/submissions.log"
  echo "$CASE: Submitted $job"
}
start_monitor() {
  export REPO
  nohup bash "$BUNDLE/scripts/watch.sh" "$CASE" >> "$RUN_DIR/logs/monitor.log" 2>&1 < /dev/null 9>&- &
  echo "Monitor: $RUN_DIR/logs/monitor.log"
}
