#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-athenak-grmhd-reference
source /lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu/environment.sh
python -u "$suite/run_bns.py" "$1" "$2"
