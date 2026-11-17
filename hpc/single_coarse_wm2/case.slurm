#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-single-coarse-convergence
source "$suite/environment.sh"
export COHORT_MODE=$1 COHORT_KIND=$2 COHORT_N=$3
python -u "$suite/run_case.py"
