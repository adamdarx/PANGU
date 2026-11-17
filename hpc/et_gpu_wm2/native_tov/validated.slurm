#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu
source "$suite/environment.sh"
export REFERENCE_N=$1 REFERENCE_MODE=$2
python -u "$suite/native-tov-reference/run_validated.py"
