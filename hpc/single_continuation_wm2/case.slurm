#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-single-continuation
source "$suite/environment.sh"
python -u "$suite/run.py" "$@"
