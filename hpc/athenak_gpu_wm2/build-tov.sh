#!/usr/bin/env bash
set -euo pipefail
suite=/lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-athenak-grmhd-reference
source /lustre/home/2601110206/CODEX_WORKPLACE/Operations/20261117-einstein-toolkit-gpu/environment.sh
export NVCC_WRAPPER_DEFAULT_COMPILER=/lustre/software/gcc/12.2.0/bin/g++
cmake -S "$suite/source" -B "$suite/build-tov" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$suite/source/kokkos/bin/nvcc_wrapper" \
  -DPROBLEM=dyn_grmhd/dyngr_tov -DAthena_ENABLE_MPI=ON -DAthena_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_AMPERE80=ON
cmake --build "$suite/build-tov" -j4
