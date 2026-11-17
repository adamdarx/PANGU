#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-single-athenak-parallel
source "$suite/environment.sh"
export OMPI_MCA_io=ompio
python -u "$suite/run_case.py" "$@"
