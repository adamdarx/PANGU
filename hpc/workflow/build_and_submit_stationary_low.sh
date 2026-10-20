#!/usr/bin/env bash
# Login node only. Build the pinned Gitee checkout and submit one frozen Stage02 run.
set -euo pipefail
expected_commit="${1:?expected PANGU commit required}"
workflow_dir="${2:?absolute workflow directory required}"
repo_dir=/lustre/home/2601110206/PANGU
dependency_prefix=/lustre/home/2601110206
partition="${PANGU_SLURM_PARTITION:-GPU40G}"
[[ "$partition" == GPU40G || "$partition" == GPU80G ]]
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$workflow_dir" == "$dependency_prefix"/pangu-workflow-* ]]
[[ "$(hostname -s)" == wm2-login* ]]
mkdir -p "$workflow_dir"
exec 9>"$workflow_dir/build-submit.lock"
flock -n 9 || { printf 'This workflow is already active\n' >&2; exit 2; }
if [[ -f "$workflow_dir/job-id.txt" ]]; then
  printf 'Already submitted: %s\n' "$(< "$workflow_dir/job-id.txt")"
  exit 0
fi
if [[ -f "$workflow_dir/submission-attempted.txt" ]]; then
  printf 'A submission was already attempted; inspect sbatch.stdout and squeue before retrying\n' >&2
  exit 2
fi
cd "$repo_dir"
bash update.sh build "$expected_commit" "$workflow_dir"
grep -q '^PARTHENON_ENABLE_HOST_COMM_BUFFERS:BOOL=OFF$' build/CMakeCache.txt
grep -q '^PARTHENON_DISABLE_SPARSE:BOOL=ON$' build/CMakeCache.txt
grep -q '^CMAKE_BUILD_TYPE:STRING=Release$' build/CMakeCache.txt
grep -q '^Kokkos_ARCH_AMPERE80:BOOL=ON$' build/CMakeCache.txt
git diff --quiet --ignore-submodules=all
test "$(git rev-parse HEAD)" = "$expected_commit"

# Preserve the original Stage02 script, helper, input and relative directory layout.
# SLURM_SUBMIT_DIR will be this independent run directory, not the live checkout.
mkdir -p "$workflow_dir/stage02/scripts" \
         "$workflow_dir/stage02/inputfile/stationary_puncture" "$workflow_dir/data"
cp --preserve=mode build/src/pangu "$workflow_dir/pangu"
cp stage02/scripts/stationary_low.slurm "$workflow_dir/stationary_low.slurm"
cp stage02/scripts/run_case.sh "$workflow_dir/stage02/scripts/run_case.sh"
cp stage02/scripts/run_case.sh "$workflow_dir/effective-run-case.sh"
cp stage02/inputfile/stationary_puncture/low.in "$workflow_dir/input.in"
cp "$workflow_dir/input.in" "$workflow_dir/stage02/inputfile/stationary_puncture/low.in"
git log -1 --format='%H %ci %s' > "$workflow_dir/source.txt"
git submodule status --recursive >> "$workflow_dir/source.txt"
export PATH="$dependency_prefix/openmpi/bin:$dependency_prefix/hdf5/bin:$PATH"
export LD_LIBRARY_PATH="$dependency_prefix/openmpi/lib:$dependency_prefix/hdf5/lib:${LD_LIBRARY_PATH:-}"
unset AWA_FAMILY AWA_VARIANT AWA_AMPLITUDE AWA_FINAL_TIME AWA_SAVE_INITIAL AWA_INTERVAL
cd "$workflow_dir"
sha256sum pangu > pangu.sha256
sha256sum input.in > input.sha256
date -Is > submission-attempted.txt
printf '%s\n' "$partition" > submission-partition.txt
# Explicit CLI partition overrides the frozen original script's GPU80G header.
submission=$(sbatch --parsable --chdir="$workflow_dir" --job-name=S02_STAT_LOW \
  --partition="$partition" \
  --comment="${workflow_dir##*/}" \
  --output="$workflow_dir/slurm-%j.out" --error="$workflow_dir/slurm-%j.err" \
  --export="ALL,PANGU_EXECUTABLE=$workflow_dir/pangu,PANGU_OUTPUT_DIR=$workflow_dir/data" \
  stationary_low.slurm)
printf '%s\n' "$submission" > sbatch.stdout
job_id="${submission%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]]
printf '%s\n' "$job_id" > job-id.txt
printf 'SUBMITTED job=%s commit=%s time=%s\n' "$job_id" "$expected_commit" "$(date -Is)"
squeue -j "$job_id" -o '%.18i %.20j %.12P %.10T %.10M %R'
