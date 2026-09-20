#!/usr/bin/env bash

set -euo pipefail

CUDA_VERSION="12.8.1"
CUDA_RUNFILE="cuda_12.8.1_570.124.06_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${CUDA_RUNFILE}"
OPENMPI_VERSION="5.0.10"
OPENMPI_ARCHIVE="openmpi-${OPENMPI_VERSION}.tar.gz"
OPENMPI_URL="https://download.open-mpi.org/release/open-mpi/v5.0/${OPENMPI_ARCHIVE}"
HDF5_VERSION="1.14.3"
HDF5_ARCHIVE="hdf5-${HDF5_VERSION}.tar.gz"
HDF5_URL="https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_3/downloads/${HDF5_ARCHIVE}"

PANGU_DEPS_ROOT="${PANGU_DEPS_ROOT:-${HOME}/.local/pangu-deps}"
PANGU_CUDA_ROOT="${PANGU_CUDA_ROOT:-${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda-12.8}}}"
PANGU_OPENMPI_ROOT="${PANGU_OPENMPI_ROOT:-${MPI_HOME:-${PANGU_DEPS_ROOT}/openmpi-${OPENMPI_VERSION}}}"
PANGU_HDF5_ROOT="${PANGU_HDF5_ROOT:-${HDF5_ROOT:-${PANGU_DEPS_ROOT}/hdf5-${HDF5_VERSION}}}"
PANGU_DEPS_BUILD_ROOT="${PANGU_DEPS_BUILD_ROOT:-${PANGU_DEPS_ROOT}/src}"
PANGU_DEPS_ENV="${PANGU_DEPS_ENV:-${PANGU_DEPS_ROOT}/pangu-env.sh}"

have_command() {
  command -v "$1" >/dev/null 2>&1
}

download() {
  local url="$1"
  local output="$2"
  if have_command curl; then
    curl -fL --retry 3 -o "${output}" "${url}"
  elif have_command wget; then
    wget -O "${output}" "${url}"
  else
    echo "Neither curl nor wget is available." >&2
    return 1
  fi
}

cuda_nvcc() {
  if [[ -x "${PANGU_CUDA_ROOT}/bin/nvcc" ]]; then
    printf '%s\n' "${PANGU_CUDA_ROOT}/bin/nvcc"
  elif have_command nvcc; then
    command -v nvcc
  fi
}

openmpi_mpirun() {
  if [[ -x "${PANGU_OPENMPI_ROOT}/bin/mpirun" ]]; then
    printf '%s\n' "${PANGU_OPENMPI_ROOT}/bin/mpirun"
  elif have_command mpirun && mpirun --version 2>/dev/null | grep -q "Open MPI"; then
    command -v mpirun
  fi
}

parallel_h5pcc() {
  local candidate=""
  if [[ -x "${PANGU_HDF5_ROOT}/bin/h5pcc" ]]; then
    candidate="${PANGU_HDF5_ROOT}/bin/h5pcc"
  elif have_command h5pcc; then
    candidate="$(command -v h5pcc)"
  fi
  if [[ -n "${candidate}" ]] &&
     "${candidate}" -showconfig 2>/dev/null | grep -Eq "Parallel HDF5:[[:space:]]+yes"; then
    printf '%s\n' "${candidate}"
  fi
}

check_dependencies() {
  local cuda=""
  local mpi=""
  local hdf5=""
  cuda="$(cuda_nvcc || true)"
  mpi="$(openmpi_mpirun || true)"
  hdf5="$(parallel_h5pcc || true)"

  echo "PANGU dependency status"
  echo "  CUDA toolkit:  ${cuda:-missing}"
  if [[ -n "${cuda}" ]]; then
    "${cuda}" --version | tail -n 1 | sed 's/^/    /'
  fi
  echo "  OpenMPI:        ${mpi:-missing}"
  if [[ -n "${mpi}" ]]; then
    "${mpi}" --version 2>/dev/null | head -n 1 | sed 's/^/    /'
  fi
  echo "  Parallel HDF5:  ${hdf5:-missing}"
  if [[ -n "${hdf5}" ]]; then
    "${hdf5}" -showconfig | grep -E "HDF5 Version:|Parallel HDF5:" | sed 's/^/    /'
  fi
}

install_cuda() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "The bundled CUDA installer workflow supports Linux only." >&2
    return 1
  fi
  mkdir -p "${PANGU_DEPS_BUILD_ROOT}"
  local runfile="${PANGU_DEPS_BUILD_ROOT}/${CUDA_RUNFILE}"
  [[ -f "${runfile}" ]] || download "${CUDA_URL}" "${runfile}"
  chmod +x "${runfile}"
  echo "Installing CUDA ${CUDA_VERSION} toolkit into ${PANGU_CUDA_ROOT}."
  echo "The NVIDIA driver is intentionally not installed or replaced."
  sudo sh "${runfile}" --silent --toolkit --toolkitpath="${PANGU_CUDA_ROOT}"
}

install_openmpi() {
  local cuda_root="${PANGU_CUDA_ROOT}"
  local nvcc=""
  nvcc="$(cuda_nvcc || true)"
  if [[ -z "${nvcc}" ]]; then
    echo "CUDA must be installed before the CUDA-aware OpenMPI build." >&2
    return 1
  fi
  cuda_root="$(cd "$(dirname "${nvcc}")/.." && pwd)"
  mkdir -p "${PANGU_DEPS_BUILD_ROOT}" "${PANGU_OPENMPI_ROOT}"
  local archive="${PANGU_DEPS_BUILD_ROOT}/${OPENMPI_ARCHIVE}"
  [[ -f "${archive}" ]] || download "${OPENMPI_URL}" "${archive}"
  tar -xzf "${archive}" -C "${PANGU_DEPS_BUILD_ROOT}"
  local source="${PANGU_DEPS_BUILD_ROOT}/openmpi-${OPENMPI_VERSION}"
  (
    cd "${source}"
    ./configure \
      --prefix="${PANGU_OPENMPI_ROOT}" \
      --with-cuda="${cuda_root}" \
      --with-cuda-libdir="${cuda_root}/lib64" \
      --enable-shared \
      --disable-static \
      --disable-mpi-fortran
    make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
    make install
  )
}

install_hdf5() {
  local mpicc="${PANGU_OPENMPI_ROOT}/bin/mpicc"
  if [[ ! -x "${mpicc}" ]]; then
    echo "OpenMPI must be installed at ${PANGU_OPENMPI_ROOT} before HDF5." >&2
    return 1
  fi
  mkdir -p "${PANGU_DEPS_BUILD_ROOT}" "${PANGU_HDF5_ROOT}"
  local archive="${PANGU_DEPS_BUILD_ROOT}/${HDF5_ARCHIVE}"
  [[ -f "${archive}" ]] || download "${HDF5_URL}" "${archive}"
  tar -xzf "${archive}" -C "${PANGU_DEPS_BUILD_ROOT}"
  local source="${PANGU_DEPS_BUILD_ROOT}/hdf5-${HDF5_VERSION}"
  (
    cd "${source}"
    CC="${mpicc}" ./configure \
      --prefix="${PANGU_HDF5_ROOT}" \
      --enable-shared \
      --enable-parallel
    make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
    make install
    make check-install
  )
}

write_environment() {
  mkdir -p "$(dirname "${PANGU_DEPS_ENV}")"
  cat >"${PANGU_DEPS_ENV}" <<EOF
export PANGU_CUDA_ROOT="${PANGU_CUDA_ROOT}"
export CUDA_HOME="${PANGU_CUDA_ROOT}"
export PANGU_OPENMPI_ROOT="${PANGU_OPENMPI_ROOT}"
export PANGU_HDF5_ROOT="${PANGU_HDF5_ROOT}"
export HDF5_ROOT="${PANGU_HDF5_ROOT}"
export PATH="${PANGU_CUDA_ROOT}/bin:${PANGU_OPENMPI_ROOT}/bin:${PANGU_HDF5_ROOT}/bin:\${PATH}"
export LD_LIBRARY_PATH="${PANGU_CUDA_ROOT}/lib64:${PANGU_OPENMPI_ROOT}/lib:${PANGU_HDF5_ROOT}/lib:\${LD_LIBRARY_PATH:-}"
EOF
  echo "Environment file written to ${PANGU_DEPS_ENV}"
  echo "Activate it with: source \"${PANGU_DEPS_ENV}\""
}

install_choice() {
  case "$1" in
    cuda) install_cuda ;;
    openmpi) install_openmpi ;;
    hdf5) install_hdf5 ;;
    all)
      [[ -n "$(cuda_nvcc || true)" ]] || install_cuda
      [[ -n "$(openmpi_mpirun || true)" ]] || install_openmpi
      [[ -n "$(parallel_h5pcc || true)" ]] || install_hdf5
      ;;
    *) echo "Unknown dependency: $1" >&2; return 2 ;;
  esac
  write_environment
}

interactive_menu() {
  check_dependencies
  while true; do
    cat <<'EOF'

Select an action:
  1) Install CUDA toolkit 12.8.1
  2) Install CUDA-aware OpenMPI 5.0.10
  3) Install Parallel HDF5 1.14.3
  4) Install all missing dependencies
  5) Recheck dependencies
  0) Exit
EOF
    read -r -p "> " choice
    case "${choice}" in
      1) install_choice cuda ;;
      2) install_choice openmpi ;;
      3) install_choice hdf5 ;;
      4) install_choice all ;;
      5) check_dependencies ;;
      0) return 0 ;;
      *) echo "Please select 0 through 5." ;;
    esac
  done
}

usage() {
  cat <<EOF
Usage: $0 [--check | --install cuda|openmpi|hdf5|all]

With no arguments, the script checks environment variables and PATH, then
opens an interactive installation menu.
EOF
}

case "${1:-}" in
  "") interactive_menu ;;
  --check) check_dependencies ;;
  --install)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    install_choice "$2"
    check_dependencies
    ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
