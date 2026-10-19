#!/usr/bin/env bash
# Install as PANGU/update.sh only after backing up the server-local original.
# bash update.sh fetch COMMIT WORKFLOW_DIR  (wm2-data)
# bash update.sh build COMMIT WORKFLOW_DIR  (wm2-login)
set -euo pipefail
mode="${1:?usage: bash update.sh fetch|build COMMIT WORKFLOW_DIR}"
expected_commit="${2:?expected commit required}"
workflow_dir="${3:?absolute workflow directory required}"
repo_dir=/lustre/home/2601110206/PANGU
driver="$repo_dir/hpc/workflow/remote_update_build.sh"
if [[ ! -f "$driver" ]]; then
  driver="$workflow_dir/remote_update_build.sh"
fi
bash "$driver" "$mode" "$expected_commit" "$workflow_dir"
