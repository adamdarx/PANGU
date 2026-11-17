#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu
source "$suite/environment.sh"
cd "$suite/Cactus"
[[ $(hostname) == wm2-login* ]]
./simfactory/bin/sim build wm2cuda --machine=generic \
  --optionlist="$suite/wm2-cuda.cfg" --thornlist="$suite/bns-reference/et-gpu-bns.th" --jobs=2
