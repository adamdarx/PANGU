#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu
source "$suite/environment.sh"
python -u "$suite/bns-reference/run.py" "$@"
