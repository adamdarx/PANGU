#!/usr/bin/env bash
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Use submit.sh}"
source ./scripts/common.sh
select_case "${1:?Missing case}"
predecessor=${2:-}
load_environment
exec 9>"$RUN_DIR/run.lock"
flock -n 9 || { echo 'Run already locked' >&2; exit 1; }
[[ ! -e "$RUN_DIR/STOP" && ! -e "$RUN_DIR/DONE" ]] || exit 0
cd "$RUN_DIR"
if [[ -e input.used.in ]]; then
  cmp -s "$INPUT" input.used.in || { echo 'Input changed; use a new run directory.' >&2; exit 1; }
else
  cp "$INPUT" input.used.in
fi
job=${SLURM_JOB_ID:?Missing Slurm allocation}
"${MPI_LAUNCH[@]}" "$EXE" --pangu-version > "logs/version-$job.txt" 2>&1
grep -q 'Metric mks' "logs/version-$job.txt"
grep -q 'Geometry mode static' "logs/version-$job.txt"
"${MPI_LAUNCH[@]}" "$EXE" --check-input -i "$INPUT" > "logs/input-$job.txt" 2>&1
"$PYTHON" "$BUNDLE/scripts/checkpoint.py" "$RUN_DIR" > "logs/candidates-$job.txt"
restart=''; restart_time=0; index=0
while IFS=$'\t' read -r saved_time path; do
  [[ -n "$path" ]] || continue
  index=$((index + 1))
  # -m constructs the restart mesh and reads conserved fields before exiting.
  if "${MPI_LAUNCH[@]}" "$EXE" -r "$path" -m 2 > "logs/preflight-$job-$index.txt" 2>&1; then
    restart=$path; restart_time=$saved_time; break
  fi
  echo "Rejected restart $path; trying older candidate."
done < "logs/candidates-$job.txt"
if [[ -z "$restart" ]]; then
  shopt -s nullglob
  old=(gr_torus_sane.restart.*.rhdf)
  if [[ -n "$predecessor" || ${#old[@]} -gt 0 ]]; then
    echo 'No restart passed PANGU validation; refusing a silent fresh start.' >&2; exit 1
  fi
fi
# XDMF time has limited precision: never mark DONE using it. Even a candidate
# labeled 30000 must let PANGU inspect its exact stored time and finish normally.
retries=0
if [[ -n "$predecessor" && -s segment.start ]]; then
  if "$PYTHON" -c 'import sys; sys.exit(float(sys.argv[1]) > float(sys.argv[2]))' "$restart_time" "$(<segment.start)"; then
    [[ ! -s no_progress.count ]] || retries=$(<no_progress.count)
    retries=$((retries + 1))
  fi
fi
printf '%s\n' "$retries" > no_progress.count
((retries < MAX_NO_PROGRESS_RESTARTS)) || { echo 'No saved progress on repeated retries.' >&2; exit 1; }
printf '%s\n' "$restart_time" > segment.start
# Preserve final checkpoint AND its XML, including the case where a newer
# numbered checkpoint was selected after a hard kill in a later segment.
final="$RUN_DIR/gr_torus_sane.restart.final.rhdf"
if [[ -e "$final" ]]; then
  archived="$RUN_DIR/gr_torus_sane.restart.saved-$job.rhdf"
  [[ ! -e "$archived" ]] || { echo 'Archive collision' >&2; exit 1; }
  mv "$final" "$archived"
  [[ ! -e "$final.xdmf" ]] || mv "$final.xdmf" "$archived.xdmf"
  [[ "$restart" != "$final" ]] || restart=$archived
fi
cmd=("${MPI_LAUNCH[@]}" "$EXE")
if [[ -n "$restart" ]]; then
  cmd+=(-r "$restart" parthenon/time/tlim=30000 parthenon/time/nlim=-1)
else
  cmd+=(-i "$INPUT")
fi
printf '%q ' "${cmd[@]}" > "logs/command-$job.txt"
printf '\n' >> "logs/command-$job.txt"
log="logs/run-$job.log"
set +e
"${cmd[@]}" > "$log" 2>&1
rc=$?
set -e
tail -n 25 "$log"
if [[ "$rc" == 0 ]] && grep -q 'Driver completed\.' "$log"; then
  # Only the current successfully completed run's footer may mark DONE.
  "$PYTHON" - "$log" <<'PY'
import re, sys
with open(sys.argv[1]) as f:
    footer = [float(m.group(1)) for line in f
              if (m := re.match(r'^time=([\d.eE+\-]+)\s+cycle=', line))]
if not footer or footer[-1] < 30000:
    sys.exit('Completed before requested tlim; stopped for inspection.')
PY
  touch DONE
elif [[ "$rc" == 1 ]] && grep -q 'Driver timed out\.  Restart to continue\.' "$log"; then
  echo 'Clean signal termination; monitor may resume after scheduler completion.'
else
  echo "Unexpected solver exit $rc; monitor stops on FAILED." >&2
  exit 1
fi
