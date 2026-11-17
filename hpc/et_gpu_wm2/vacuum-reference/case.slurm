#!/usr/bin/env bash
set -euo pipefail
root=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu
source "$root/environment.sh"
python -u "$root/vacuum-reference/run.py" "$@"
