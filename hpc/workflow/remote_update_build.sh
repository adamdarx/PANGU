#!/usr/bin/env bash
# Run with bash; preserve the server's local setup and pre-existing results.
set -euo pipefail
mode="${1:?fetch or build required}"
expected_commit="${2:?expected PANGU commit required}"
workflow_dir="${3:?absolute workflow directory required}"
repo_dir=/lustre/home/2601110206/PANGU
dependency_prefix=/lustre/home/2601110206
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]]
case "$workflow_dir" in
  "$dependency_prefix"/pangu-workflow-* | \
  "$dependency_prefix"/CODEX_WORKPLACE/ApplesWithApples/deployments/*) ;;
  *) printf 'Unsupported deployment record directory: %s\n' "$workflow_dir" >&2; exit 2 ;;
esac
[[ "$workflow_dir" != *'/../'* && "$workflow_dir" != */.. ]]
mkdir -p "$workflow_dir"
cd "$repo_dir"
test -f update.sh
test -f build.sh
case "$mode:$(hostname -s)" in
  fetch:wm2-data*) ;;
  build:wm2-login*) ;;
  *) printf 'Use fetch on wm2-data and build on wm2-login\n' >&2; exit 2 ;;
esac
git diff --quiet --ignore-submodules=all
git diff --cached --quiet --ignore-submodules=all
[[ "$(git remote get-url origin)" == https://gitee.com/adamdarx/PANGU* ]]
if [[ ! -f "$workflow_dir/original-update.sh" ]]; then
  cp --preserve=mode,timestamps update.sh "$workflow_dir/original-update.sh"
fi
if [[ ! -f "$workflow_dir/original-build.sh" ]]; then
  cp --preserve=mode,timestamps build.sh "$workflow_dir/original-build.sh"
fi
if [[ "$mode" == fetch ]]; then
git rev-parse HEAD > "$workflow_dir/pangu-before.txt"
git -c fetch.recurseSubmodules=false fetch --no-recurse-submodules origin main
test "$(git rev-parse origin/main)" = "$expected_commit"

# The known server-only boundary experiment must not be silently overwritten
# or mixed into the pinned framework revision. Keep both a patch and a stash.
if ! git -C parthenon diff --quiet --ignore-submodules=all; then
  git -C parthenon diff --binary > "$workflow_dir/parthenon-server-changes.patch"
  git -C parthenon stash push -m "Preserve server changes before $expected_commit"
  git -C parthenon rev-parse refs/stash > "$workflow_dir/parthenon-stash.txt"
fi
git -C parthenon diff --quiet --ignore-submodules=all
git -C parthenon diff --cached --quiet --ignore-submodules=all
git -c submodule.recurse=false merge --ff-only "$expected_commit"
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive > "$workflow_dir/submodules.txt"
if grep -Eq '^[-+U]' "$workflow_dir/submodules.txt"; then
  printf 'Submodule checkout does not match the recorded gitlinks\n' >&2
  exit 1
fi
printf 'FETCH_OK %s %s\n' "$expected_commit" "$(date -Is)"
exit 0
fi

# Use the user's CUDA-aware MPI and parallel HDF5, not the non-CUDA site MPI.
test "$(git rev-parse HEAD)" = "$expected_commit"
git submodule status --recursive > "$workflow_dir/submodules.txt"
if grep -Eq '^[-+U]' "$workflow_dir/submodules.txt"; then
  printf 'Run the data-node fetch phase before building\n' >&2
  exit 1
fi
module load gcc/12.2.0 cuda/12.6.0 cmake/3.31.9
export PATH="$dependency_prefix/openmpi/bin:$dependency_prefix/hdf5/bin:$PATH"
export LD_LIBRARY_PATH="$dependency_prefix/openmpi/lib:$dependency_prefix/hdf5/lib:${LD_LIBRARY_PATH:-}"
export HDF5_ROOT="$dependency_prefix/hdf5"
export CMAKE_PREFIX_PATH="$dependency_prefix/openmpi:$dependency_prefix/hdf5:${CMAKE_PREFIX_PATH:-}"
printf 'UPDATE_BUILD_BEGIN %s\n' "$(date -Is)"
bash build.sh
test "$(git rev-parse HEAD)" = "$expected_commit"
test -x build/src/pangu
git diff --quiet --ignore-submodules=all
git submodule status --recursive
cp build/CMakeCache.txt "$workflow_dir/CMakeCache.txt"
sha256sum build/src/pangu > "$workflow_dir/pangu.sha256"
ldd build/src/pangu > "$workflow_dir/pangu.ldd.txt"
while read -r missing_library; do
  [[ -z "$missing_library" ]] && continue
  if [[ "$missing_library" == libcuda.so.1 ]]; then
    # Login nodes do not provide the GPU driver. Validate it in the Slurm
    # allocation, not by running the CUDA executable on the login node.
    printf 'GPU driver check deferred to allocated GPU node: %s\n' "$missing_library"
  else
    printf 'Unresolved non-driver runtime dependency: %s\n' "$missing_library" >&2
    exit 1
  fi
done < <(awk '$2 == "=>" && $3 == "not" && $4 == "found" {print $1}' \
            "$workflow_dir/pangu.ldd.txt")
printf 'UPDATE_BUILD_OK %s %s\n' "$expected_commit" "$(date -Is)"
