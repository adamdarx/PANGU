#!/usr/bin/env bash
# Run with bash; preserve the server's local setup and pre-existing results.
set -euo pipefail
expected_commit="${1:?expected PANGU commit required}"
workflow_dir="${2:?absolute workflow directory required}"
repo_dir=/lustre/home/2601110206/PANGU
dependency_prefix=/lustre/home/2601110206
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$workflow_dir" == "$dependency_prefix"/pangu-workflow-* ]]
mkdir -p "$workflow_dir"
cd "$repo_dir"
test -f update.sh
test -f build.sh
git diff --quiet --ignore-submodules=all
git diff --cached --quiet --ignore-submodules=all
[[ "$(git remote get-url origin)" == https://gitee.com/adamdarx/PANGU* ]]
cp --preserve=mode,timestamps update.sh "$workflow_dir/original-update.sh"
cp --preserve=mode,timestamps build.sh "$workflow_dir/original-build.sh"
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

# Use the user's CUDA-aware MPI and parallel HDF5, not the non-CUDA site MPI.
export PATH="$dependency_prefix/openmpi/bin:$dependency_prefix/hdf5/bin:$PATH"
export LD_LIBRARY_PATH="$dependency_prefix/openmpi/lib:$dependency_prefix/hdf5/lib:${LD_LIBRARY_PATH:-}"
export HDF5_ROOT="$dependency_prefix/hdf5"
export CMAKE_PREFIX_PATH="$dependency_prefix/openmpi:$dependency_prefix/hdf5:${CMAKE_PREFIX_PATH:-}"
printf 'UPDATE_BUILD_BEGIN %s\n' "$(date -Is)"
bash update.sh
test "$(git rev-parse HEAD)" = "$expected_commit"
test -x build/src/pangu
git diff --quiet --ignore-submodules=all
git submodule status --recursive
cp build/CMakeCache.txt "$workflow_dir/CMakeCache.txt"
sha256sum build/src/pangu > "$workflow_dir/pangu.sha256"
ldd build/src/pangu > "$workflow_dir/pangu.ldd.txt"
if grep -q 'not found' "$workflow_dir/pangu.ldd.txt"; then
  printf 'Unresolved runtime dependency\n' >&2
  exit 1
fi
printf 'UPDATE_BUILD_OK %s %s\n' "$expected_commit" "$(date -Is)"
