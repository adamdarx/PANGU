#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_environment
cmake -S "$REPO" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$REPO/parthenon/external/Kokkos/bin/nvcc_wrapper" \
  -DCMAKE_C_COMPILER=mpicc -DHDF5_PREFER_PARALLEL=ON \
  -DPANGU_ENABLE_CUDA=ON -DPANGU_ENABLE_MPI=ON -DPANGU_ENABLE_HDF5=ON \
  -DPANGU_ENABLE_OPENMP=OFF -DPANGU_ENABLE_TESTING=OFF -DBUILD_TESTING=OFF \
  -DPHYSICS=gr -DMETRIC=mks -DMODE=static -DESTIMATOR=wave \
  -DKokkos_ARCH_${GPU_ARCH}=ON -DPARTHENON_ENABLE_HOST_COMM_BUFFERS=ON \
  -DPANGU_SINGLE_PRECISION=OFF
cmake --build "$BUILD_DIR" --target pangu -j "$BUILD_JOBS"
echo "Built $EXE. Runtime checks take place only in the GPU allocation."
