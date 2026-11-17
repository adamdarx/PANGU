#!/usr/bin/env bash
set -euo pipefail
source /lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu/environment.sh
python -u /lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-athenak-grmhd-reference/tov-repair/run_tov.py "$@"
