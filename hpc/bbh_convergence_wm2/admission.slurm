#!/usr/bin/env bash
#SBATCH --job-name=BBH_QC_RES_ADMIT
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-bbh-convergence/logs/admit.%j.out
#SBATCH --error=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-bbh-convergence/logs/admit.%j.err
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-bbh-convergence
source "$suite/environment.sh"
python -u "$suite/resource_gate.py"
