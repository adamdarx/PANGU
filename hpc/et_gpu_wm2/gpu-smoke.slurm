#!/usr/bin/env bash
#SBATCH --job-name=ET_CarpetX_AsterX_GPU_SMOKE
#SBATCH --partition=GPU80G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu/logs/smoke.%j.out
#SBATCH --error=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu/logs/smoke.%j.err
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu
source "$suite/environment.sh"
python -u "$suite/gpu_smoke.py"
